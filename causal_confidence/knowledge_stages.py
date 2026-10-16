"""Reporter training and evaluation across knowledge variants.

Pipeline (revised protocol steps 3-4):

1. Collect states: for every (question, variant), retain prefix probability
   features, rich logits, and the hidden state, plus an outcome bank from the
   frozen answer policy.  Reporter inputs never include the variant id,
   exposure list, gold answer, or sampled answer text.
2. Fit the shared reporter with the pointwise objective (T1) and the same
   example bank plus the within-question pair objective (T2); lambda is
   selected on the development split only.
3. Evaluate on held-out subjects (seen variants) and on a held-out knowledge
   adapter (variant 5, its own subjects) with the probability baseline and the
   incremental-Brier test.

The confidence head is a grid-mean MLP over state features (feasibility
family).  A native adapter is a later, separate step and is not implemented
here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import (
    BaselineFeatures,
    baseline_predictions,
    fit_predict_augmented,
    hidden_features,
    prefix_probability_features,
    question_features,
    rich_prefix_features,
    select_strongest_baseline,
)
from .data import Question
from .episodes import generate_answers
from .evaluate import (
    ClusterItem,
    calibration_slope_intercept,
    cluster_bootstrap,
    expected_calibration_error,
)
from .objectives import grid_moments, grid_tensor
from .outcomes import match_answer
from .reporters import GridReportHead


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------
@dataclass
class StateData:
    fact_ids: list[str]
    subject_ids: list[str]
    variant_ids: list[int]
    features: BaselineFeatures
    outcomes: np.ndarray

    @property
    def success(self) -> np.ndarray:
        return self.outcomes.mean(axis=1)

    @property
    def n_states(self) -> int:
        return len(self.fact_ids)

    def reporter_matrix(self) -> np.ndarray:
        return self.features.stack(["question", "prefix", "hidden"])


def collect_states(
    env,
    facts: list,
    variant_adapters: dict[int, list | None],
    *,
    template: str = "eval0",
    draws: int = 8,
    seed: int = 0,
    partition: str = "train",
    progress_every: int = 50,
) -> StateData:
    """Collect states for several variants using already-installed adapters.

    Prefer :func:`collect_all_variants`, which installs one checkpoint at a
    time and removes it afterwards.
    """

    collected = []
    for variant, adapters in sorted(variant_adapters.items()):
        if adapters is not None:
            raise ValueError("collect_states expects adapters already installed or None")
        item = collect_states_for_variant(
            env,
            facts,
            variant_id=variant,
            template=template,
            draws=draws,
            seed=seed,
            partition=partition,
        )
        collected.append(item)
    return merge_variant_states(collected)


class installed_variant:
    """Context manager installing one variant's checkpoints (or the base model)."""

    def __init__(self, env, *, layers, rank, state_path=None, variant_id=0):
        self.env = env
        self.layers = layers
        self.rank = rank
        self.state_path = state_path
        self.variant_id = variant_id
        self.adapters = None

    def __enter__(self):
        from .variants import install_variant

        if self.state_path is not None:
            self.adapters = install_variant(
                self.env, layers=self.layers, rank=self.rank, state_path=self.state_path
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        from .variants import remove_variant

        if self.adapters is not None:
            remove_variant(self.env, self.adapters)
            self.adapters = None
        return None


@torch.no_grad()
def collect_states_for_variant(
    env,
    facts: list,
    *,
    variant_id: int,
    template: str,
    draws: int,
    seed: int,
    partition: str,
) -> dict:
    """Retained-state features and outcome banks for one variant."""

    result = {"fact_ids": [], "subject_ids": [], "question": [], "prefix": [], "rich": [], "hidden": [], "outcomes": []}
    for index, fact in enumerate(facts):
        question = question_from_fact(fact, template)
        ids, digest = env.prefix_ids(question)
        from .episodes import run_prefix, suffix_state

        prefix = run_prefix(env.model, ids, capture_hidden=False, prompt_hash_value=digest)
        # The answer fork is after the suffix, not at the end of the question.
        fork_logits, fork_hidden = suffix_state(env.model, prefix, env.answer_suffix)
        draws_out = generate_answers(
            env.model,
            prefix,
            env.answer_suffix,
            env.tokenizer,
            draws=draws,
            base_seed=seed * 1_000_003 + index * 31 + variant_id * 7,
        )
        matched = np.asarray(
            [match_answer(draw.text, list(fact.references)) for draw in draws_out],
            dtype=np.float32,
        )
        result["fact_ids"].append(fact.fact_id)
        result["subject_ids"].append(fact.subject_id)
        result["outcomes"].append(matched)
        result["question"].append(
            question_features([fact.attribute], [None], [question.question_text])
        )
        logits = fork_logits[0].float().cpu().numpy()
        result["prefix"].append(prefix_probability_features(logits[None, :]))
        result["rich"].append(rich_prefix_features(logits[None, :]))
        # Keep the raw final hidden state, not a lossy random projection: the
        # readout audit showed the projection destroyed usable signal.
        hidden = fork_hidden[0].float().cpu().numpy()
        result["hidden"].append(hidden.astype(np.float32))
    stacked = {
        key: np.concatenate(values, axis=0).astype(np.float32)
        for key, values in result.items()
        if key in ("question", "prefix", "rich", "hidden")
    }
    return {
        "variant_id": variant_id,
        "fact_ids": result["fact_ids"],
        "subject_ids": result["subject_ids"],
        "features": BaselineFeatures(blocks=stacked),
        "outcomes": np.stack(result["outcomes"]).astype(np.float32),
        "partition": partition,
        "template": template,
    }


def question_from_fact(fact, template: str = "eval0") -> Question:
    if template == "train":
        text = fact.train_question
    else:
        text = fact.eval_questions[int(template.replace("eval", ""))]
    return Question(
        question_id=fact.fact_id,
        subject_id=fact.subject_id,
        subject_name=fact.subject_name,
        relation=fact.attribute,
        question_text=text,
        references=fact.references,
        source="knowledge",
    )


def merge_variant_states(collected: list[dict]) -> StateData:
    fact_ids: list[str] = []
    subject_ids: list[str] = []
    variant_ids: list[int] = []
    blocks: dict[str, list[np.ndarray]] = {"question": [], "prefix": [], "rich": [], "hidden": []}
    outcomes: list[np.ndarray] = []
    for item in collected:
        fact_ids.extend(item["fact_ids"])
        subject_ids.extend(item["subject_ids"])
        variant_ids.extend([item["variant_id"]] * len(item["fact_ids"]))
        for key in blocks:
            blocks[key].append(item["features"].blocks[key])
        outcomes.append(item["outcomes"])
    return StateData(
        fact_ids=fact_ids,
        subject_ids=subject_ids,
        variant_ids=variant_ids,
        features=BaselineFeatures(
            {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in blocks.items()}
        ),
        outcomes=np.concatenate(outcomes, axis=0).astype(np.float32),
    )


def collect_all_variants(
    env,
    *,
    layers: tuple[int, ...],
    rank: int,
    variant_states: dict[int, str | None],
    facts: list,
    template: str,
    draws: int,
    seed: int,
    partition: str,
    progress_every: int = 50,
) -> StateData:
    collected = []
    for variant, state_path in sorted(variant_states.items()):
        with installed_variant(env, layers=layers, rank=rank, state_path=state_path, variant_id=variant):
            item = collect_states_for_variant(
                env,
                facts,
                variant_id=variant,
                template=template,
                draws=draws,
                seed=seed,
                partition=partition,
            )
        collected.append(item)
        print(
            f"    collected variant {variant}: {len(item['fact_ids'])} states, "
            f"mean success {item['outcomes'].mean():.3f}",
            flush=True,
        )
    return merge_variant_states(collected)


# ---------------------------------------------------------------------------
# Reporter fitting
# ---------------------------------------------------------------------------
def question_groups(data: StateData) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, fact_id in enumerate(data.fact_ids):
        groups.setdefault(fact_id, []).append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def state_groups(data: StateData) -> list[list[int]]:
    """One group per fact; used when a split has a single variant per question."""

    groups: dict[str, list[int]] = {}
    for index, fact_id in enumerate(data.fact_ids):
        groups.setdefault(fact_id, []).append(index)
    return list(groups.values())


def evaluation_groups(data: StateData) -> list[list[int]]:
    """Multi-variant question groups when available, per-fact groups otherwise."""

    groups = question_groups(data)
    return groups if groups else state_groups(data)


class StandardizedReporter:
    """Grid reporter over standardized features.

    Raw reporter features mix small probabilities with a large-scale hidden
    block; an unstandardized MLP overfits the hidden noise (dev Brier near the
    constant baseline).  Feature means and scales are fixed from the training
    states, matching the baseline pipelines' StandardScaler.
    """

    def __init__(self, head: GridReportHead, mean: np.ndarray, std: np.ndarray):
        self.head = head
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def expected(self, x) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return self.head.expected((tensor - self.mean) / self.std)


def fit_pair_reporter(
    data: StateData,
    *,
    lam: float = 0.0,
    steps: int = 1_500,
    lr: float = 1e-3,
    batch_questions: int = 48,
    hidden: int = 256,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> StandardizedReporter:
    """Grid-mean reporter trained with the pointwise + pair objective.

    ``lam=0`` is the T1 control; ``lam>0`` adds the within-question contrast
    term with equal weight per question and per variant pair.  Inputs are
    standardized with training statistics.
    """

    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    raw = data.reporter_matrix().astype(np.float32)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0) + 1e-6
    x = torch.tensor((raw - mean) / std, dtype=torch.float32)
    targets = torch.tensor(data.success, dtype=torch.float32)
    groups = question_groups(data)
    if not groups:
        raise ValueError("reporter training needs at least two variants per question")
    n_variants = len(groups[0])
    if any(len(group) != n_variants for group in groups):
        raise ValueError("every question must have the same number of variant states")
    torch.manual_seed(seed)
    head = GridReportHead(x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    grid = grid_tensor()
    for _ in range(steps):
        chosen = torch.randint(len(groups), (min(batch_questions, len(groups)),), generator=generator)
        indices: list[int] = []
        for position in chosen.tolist():
            indices.extend(groups[position])
        index = torch.tensor(indices, dtype=torch.long)
        probs = head(x[index])
        mean_value, _ = grid_moments(probs, grid)
        point = (mean_value - targets[index]).pow(2).mean()
        loss = point
        if lam > 0:
            n_questions = len(indices) // n_variants
            mu = mean_value.reshape(n_questions, n_variants)
            target = targets[index].reshape(n_questions, n_variants)
            mu_pairs = mu[:, :, None] - mu[:, None, :]
            target_pairs = target[:, :, None] - target[:, None, :]
            pair = (mu_pairs - target_pairs).pow(2).mean()
            loss = point + lam * pair
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.eval()
    return StandardizedReporter(head, mean, std)


def reporter_objective(head, data: StateData, *, lam: float) -> dict:
    """Point risk, contrast risk, and their ``lam``-weighted total on ``data``.

    The pair term is always computed; ``lam`` only scales it in ``total`` so
    development selection can compare heads with one fixed weight.
    """

    mean = torch.tensor(head.expected(data.reporter_matrix()), dtype=torch.float32)
    targets = torch.tensor(data.success, dtype=torch.float32)
    point = float((mean - targets).pow(2).mean())
    groups = question_groups(data)
    pair = float("nan")
    if groups:
        n_variants = len(groups[0])
        index = torch.tensor([i for group in groups for i in group], dtype=torch.long)
        mu = mean[index].reshape(len(groups), n_variants)
        target = targets[index].reshape(len(groups), n_variants)
        mu_pairs = mu[:, :, None] - mu[:, None, :]
        target_pairs = target[:, :, None] - target[:, None, :]
        pair = float((mu_pairs - target_pairs).pow(2).mean())
    return {"point": point, "pair": pair, "total": point + lam * pair}


def select_lambda(
    train: StateData,
    dev: StateData,
    *,
    lambdas: tuple[float, ...] = (0.25, 1.0, 4.0),
    steps: int,
    seed: int,
    selection_lambda: float = 1.0,
) -> dict:
    """T1 (lambda=0) always; every candidate is trained, then scored on dev.

    Selection uses a fixed dev criterion (``point + 1.0 * pair``) so heads
    trained with different lambda values are compared on the same scale.
    """

    t1 = fit_pair_reporter(train, lam=0.0, steps=steps, seed=seed)
    results = {
        "0": {
            **reporter_objective(t1, dev, lam=selection_lambda),
            "trained_lambda": 0.0,
        }
    }
    heads = {"0": t1}
    for lam in lambdas:
        head = fit_pair_reporter(train, lam=lam, steps=steps, seed=seed)
        results[str(lam)] = {
            **reporter_objective(head, dev, lam=selection_lambda),
            "trained_lambda": lam,
        }
        heads[str(lam)] = head
    # T2 always uses a nonzero lambda: lambda=0 is the T1 control arm, not a
    # selectable pair weight (plan section 8).
    candidates = [key for key in results if key != "0"]
    best = min(candidates, key=lambda key: results[key]["total"])
    return {"scores": results, "best_lambda": float(best), "heads": heads}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def reporter_predictions(head: GridReportHead, data: StateData) -> np.ndarray:
    x = torch.tensor(data.reporter_matrix(), dtype=torch.float32)
    return head.expected(x).numpy()


def per_question_brier(
    predictions: np.ndarray, data: StateData
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw-level Brier per question, plus within-question contrast error.

    The Brier target is every independently graded answer draw, not the
    observed success fraction of a state: ``mean_k (c - Y_k)^2``.  Contrast
    error still compares report differences with measured success differences
    across variants of the same question.
    """

    groups = evaluation_groups(data)
    subjects: list[str] = []
    values: list[float] = []
    contrast_values: list[float] = []
    for group in groups:
        subjects.append(data.subject_ids[group[0]])
        error = (predictions[group][:, None] - data.outcomes[group]) ** 2
        values.append(float(error.mean()))
        if len(group) > 1:
            report_pairs = predictions[group][:, None] - predictions[group][None, :]
            success_pairs = data.success[group][:, None] - data.success[group][None, :]
            contrast_values.append(float(((report_pairs - success_pairs) ** 2).mean()))
    return subjects, np.asarray(values), np.asarray(contrast_values)


def _nanmean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def evaluate_reporter(head: GridReportHead, data: StateData, *, label: str) -> dict:
    predictions = reporter_predictions(head, data)
    subjects, briers, contrasts = per_question_brier(predictions, data)
    outcomes = data.outcomes
    return {
        "label": label,
        "n_states": int(data.n_states),
        "n_questions": len(subjects),
        "brier": float(briers.mean()),
        "brier_vs_success_mean": float(
            ((predictions - data.success) ** 2).mean()
        ),
        "contrast_error": _nanmean_or_none(contrasts),
        "ece": expected_calibration_error(predictions, outcomes),
        "calibration": calibration_slope_intercept(predictions, outcomes),
        "mean_report": float(predictions.mean()),
        "mean_success": float(data.success.mean()),
        "per_question_brier": briers.tolist(),
        "subjects": subjects,
    }


def out_of_fold_report_predictions(
    data: StateData,
    *,
    lam: float,
    steps: int,
    seed: int,
    folds: int = 4,
) -> np.ndarray:
    """Subject-grouped out-of-fold reporter predictions for ``data``.

    Second-stage predictors that include the report as an input must be fit on
    predictions the reporter did not see during training; otherwise they
    overtrust an overfitted reporter.  Folds keep every state of a subject
    (and therefore of a question) together.
    """

    subjects = list(dict.fromkeys(data.subject_ids))
    generator = np.random.default_rng(seed)
    generator.shuffle(subjects)
    fold_of = {subject: index % folds for index, subject in enumerate(subjects)}
    predictions = np.full(data.n_states, np.nan, dtype=np.float64)
    for fold in range(folds):
        train_index = [i for i, s in enumerate(data.subject_ids) if fold_of[s] != fold]
        test_index = [i for i, s in enumerate(data.subject_ids) if fold_of[s] == fold]
        if not train_index or not test_index:
            continue
        subset = StateData(
            fact_ids=[data.fact_ids[i] for i in train_index],
            subject_ids=[data.subject_ids[i] for i in train_index],
            variant_ids=[data.variant_ids[i] for i in train_index],
            features=BaselineFeatures(
                {key: value[train_index] for key, value in data.features.blocks.items()}
            ),
            outcomes=data.outcomes[train_index],
        )
        head = fit_pair_reporter(subset, lam=lam, steps=steps, seed=seed + fold)
        fold_states = StateData(
            fact_ids=[data.fact_ids[i] for i in test_index],
            subject_ids=[data.subject_ids[i] for i in test_index],
            variant_ids=[data.variant_ids[i] for i in test_index],
            features=BaselineFeatures(
                {key: value[test_index] for key, value in data.features.blocks.items()}
            ),
            outcomes=data.outcomes[test_index],
        )
        predictions[test_index] = reporter_predictions(head, fold_states)
    if np.isnan(predictions).any():
        raise ValueError("out-of-fold predictions did not cover every state")
    return predictions


def _paired_bootstrap(
    subjects: list[str], difference: np.ndarray, *, seed: int
) -> dict:
    differences = np.asarray(difference, dtype=np.float64)
    if differences.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    payload = [
        ClusterItem(subject_id=subject, payload={"value": float(value)})
        for subject, value in zip(subjects, differences)
    ]
    result = cluster_bootstrap(
        payload,
        lambda index, items: float(np.mean([items[i].payload["value"] for i in index])),
        seed=seed,
    )
    return {"mean": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}


def baseline_and_added_value(
    train: StateData,
    dev: StateData,
    test: StateData,
    report_train: np.ndarray | None,
    report_test: np.ndarray,
    *,
    seed: int,
    report_lambda: float = 1.0,
    report_steps: int = 1_200,
    folds: int = 4,
) -> dict:
    """Strongest ordinary baseline on dev; fitted added-value test on test.

    ``report_train`` must be out-of-fold (or otherwise held out) reporter
    predictions.  When it is ``None`` they are computed here with
    :func:`out_of_fold_report_predictions`.  Both Brier scores use individual
    answer draws as targets.
    """

    if report_train is None:
        report_train = out_of_fold_report_predictions(
            train, lam=report_lambda, steps=report_steps, seed=seed, folds=folds
        )
    train_features = BaselineFeatures(
        {key: value for key, value in train.features.blocks.items() if key != "rich"}
    )
    dev_features = BaselineFeatures(
        {key: value for key, value in dev.features.blocks.items() if key != "rich"}
    )
    test_features = BaselineFeatures(
        {key: value for key, value in test.features.blocks.items() if key != "rich"}
    )
    predictions = baseline_predictions(train_features, train.outcomes, dev_features, seed=seed)
    strongest, dev_scores = select_strongest_baseline(predictions, dev.outcomes)
    # Constants have no feature block to refit with; fall back to the
    # question-metadata baseline for the added-value comparison.
    if strongest in ("constant_rate", "constant_half"):
        strongest = "question"
    baseline_test = baseline_predictions(
        train_features, train.outcomes, test_features, seed=seed, include=(strongest,)
    )[strongest]
    augmented_test = fit_predict_augmented(
        train_features,
        train.outcomes,
        test_features,
        report_train,
        report_test,
        baseline=strongest,
        seed=seed,
    )
    baseline_brier = float(((baseline_test[:, None] - test.outcomes) ** 2).mean())
    augmented_brier = float(((augmented_test[:, None] - test.outcomes) ** 2).mean())
    # Per-question differences clustered by subject; fall back to one group per
    # fact when the split has a single variant per question (transfer).
    groups = evaluation_groups(test)
    per_question = []
    per_subject = []
    for group in groups:
        per_question.append(
            float(
                ((baseline_test[group][:, None] - test.outcomes[group]) ** 2).mean()
                - ((augmented_test[group][:, None] - test.outcomes[group]) ** 2).mean()
            )
        )
        per_subject.append(test.subject_ids[group[0]])
    added = {
        "baseline_brier": baseline_brier,
        "augmented_brier": augmented_brier,
        "added": baseline_brier - augmented_brier,
        "ci": _paired_bootstrap(per_subject, np.asarray(per_question), seed=seed),
    }
    return {
        "strongest_baseline": strongest,
        "dev_scores": dev_scores,
        "added_value": added,
        "baseline_test_predictions": baseline_test.tolist(),
    }


def compare_reporters(
    t1: GridReportHead, t2: GridReportHead, data: StateData, *, seed: int
) -> dict:
    predictions_t1 = reporter_predictions(t1, data)
    predictions_t2 = reporter_predictions(t2, data)
    subjects, brier_t1, contrast_t1 = per_question_brier(predictions_t1, data)
    _, brier_t2, contrast_t2 = per_question_brier(predictions_t2, data)
    contrast_t1_value = _nanmean_or_none(contrast_t1)
    contrast_t2_value = _nanmean_or_none(contrast_t2)
    contrast_reduction = None
    contrast_ci = None
    if contrast_t1_value is not None and contrast_t1_value > 0:
        if contrast_t2_value is not None:
            contrast_reduction = (contrast_t1_value - contrast_t2_value) / contrast_t1_value
        if contrast_t1.shape == contrast_t2.shape and contrast_t1.size:
            contrast_ci = _paired_bootstrap(
                subjects, contrast_t1 - contrast_t2, seed=seed
            )
    return {
        "brier_t1": float(brier_t1.mean()),
        "brier_t2": float(brier_t2.mean()),
        "delta_brier": float(brier_t1.mean() - brier_t2.mean()),
        "delta_brier_ci": _paired_bootstrap(subjects, brier_t1 - brier_t2, seed=seed),
        "contrast_t1": contrast_t1_value,
        "contrast_t2": contrast_t2_value,
        "contrast_reduction": contrast_reduction,
        "contrast_ci": contrast_ci,
    }


def write_json(path: str | Path, payload: dict) -> str:
    from .outcomes import write_immutable

    return write_immutable(path, payload)


# ---------------------------------------------------------------------------
# Full feasibility study for one seed
# ---------------------------------------------------------------------------
@dataclass
class StudyConfig:
    seed: int = 0
    subjects: int = 60
    transfer_subjects: int = 15
    train_subjects: int = 36
    dev_subjects: int = 12
    epochs: int = 40
    batch: int = 8
    rank: int = 8
    lr: float = 1e-4
    adapter_layers: tuple[int, ...] = ()
    train_draws: int = 8
    eval_draws: int = 16
    reporter_steps: int = 1_200
    lambda_grid: tuple[float, ...] = (0.25, 1.0, 4.0)
    reuse_variants: bool = True


def run_knowledge_study(env, out_dir: str | Path, config: StudyConfig) -> dict:
    """Train variants, verify the manipulation, train and evaluate reporters."""

    from . import knowledge as know
    from . import variants as var

    out_dir = Path(out_dir)
    (out_dir / "variants").mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    n_layers = int(env.model.config.num_hidden_layers)
    layers = config.adapter_layers or var.default_adapter_layers(n_layers)
    print(f"[study {config.seed}] layers={layers}", flush=True)

    facts = know.build_corpus(config.subjects, seed=config.seed, prefix="main")
    transfer_facts = know.build_corpus(
        config.transfer_subjects, seed=config.seed + 77, prefix="transfer"
    )
    exposure = know.assign_exposure(facts)
    splits = know.split_by_subject(
        facts, train_subjects=config.train_subjects, dev_subjects=config.dev_subjects
    )
    manifest = know.corpus_manifest(facts, exposure, splits, transfer_facts, seed=config.seed)
    know.write_corpus_manifest(out_dir / "manifests" / "knowledge_corpus.json", manifest)

    # --- train or load variants -------------------------------------------------
    state_paths: dict[int, str] = {}
    training_metrics: dict[str, dict] = {}
    for variant in know.VARIANT_IDS:
        path = out_dir / "variants" / f"variant_{variant}.pt"
        if config.reuse_variants and path.exists():
            state_paths[variant] = str(path)
            print(f"[study {config.seed}] variant {variant}: loaded {path.name}", flush=True)
            continue
        variant_facts = [
            fact for fact in facts if fact.fact_id in set(exposure[variant])
        ]
        print(
            f"[study {config.seed}] variant {variant}: training on {len(variant_facts)} facts",
            flush=True,
        )
        adapters, metrics = var.train_variant(
            env,
            variant_facts,
            layers=layers,
            rank=config.rank,
            epochs=config.epochs,
            batch=config.batch,
            lr=config.lr,
            seed=config.seed * 100 + variant,
        )
        var.save_variant(adapters, path)
        var.remove_variant(env, adapters)
        state_paths[variant] = str(path)
        training_metrics[str(variant)] = metrics
    transfer_path = out_dir / "variants" / f"variant_{know.TRANSFER_VARIANT_ID}.pt"
    if config.reuse_variants and transfer_path.exists():
        state_paths[know.TRANSFER_VARIANT_ID] = str(transfer_path)
        print(f"[study {config.seed}] transfer variant: loaded", flush=True)
    else:
        print(
            f"[study {config.seed}] transfer variant: training on {len(transfer_facts)} facts",
            flush=True,
        )
        adapters, metrics = var.train_variant(
            env,
            transfer_facts,
            layers=layers,
            rank=config.rank,
            epochs=config.epochs,
            batch=config.batch,
            lr=config.lr,
            seed=config.seed * 100 + know.TRANSFER_VARIANT_ID,
        )
        var.save_variant(adapters, transfer_path)
        var.remove_variant(env, adapters)
        state_paths[know.TRANSFER_VARIANT_ID] = str(transfer_path)
        training_metrics[str(know.TRANSFER_VARIANT_ID)] = metrics

    # --- verification ------------------------------------------------------------
    eval_facts = splits["dev"] + splits["test"]
    success_by_variant: dict[int, dict[str, float]] = {}
    format_stats: dict[int, dict[str, float]] = {}
    train_template_success: dict[int, dict[str, float]] = {}
    reproducibility: dict[int, float | None] = {}
    known_by_variant: dict[int, set[str]] = {
        variant: set(exposure[variant]) for variant in know.VARIANT_IDS
    }
    for variant in know.VARIANT_IDS:
        with installed_variant(
            env, layers=layers, rank=config.rank, state_path=state_paths[variant], variant_id=variant
        ):
            result = var.evaluate_facts(
                env,
                eval_facts,
                template=var.HELD_OUT_TEMPLATE,
                draws=max(4, config.eval_draws // 2),
                seed=config.seed + variant,
                partition="verification",
            )
            success_by_variant[variant] = result["success"]
            format_stats[variant] = {
                "empty_rate": result["empty_rate"],
                "truncation_rate": result["truncation_rate"],
            }
            known = sorted(
                fact_id for fact_id in known_by_variant[variant] if fact_id in result["success"]
            )[:16]
            known_facts = [fact for fact in eval_facts if fact.fact_id in set(known)]
            train_result = var.evaluate_facts(
                env,
                known_facts,
                template="train",
                draws=max(4, config.eval_draws // 2),
                seed=config.seed + 500 + variant,
                partition="verification_train_template",
            )
            train_template_success[variant] = train_result["success"]
            repeat = var.evaluate_facts(
                env,
                eval_facts[::2],
                template=var.HELD_OUT_TEMPLATE,
                draws=4,
                seed=config.seed + 900 + variant,
                partition="verification_repeat",
            )
            pairs = [
                (result["success"][fact_id], repeat["success"][fact_id])
                for fact_id in repeat["success"]
            ]
            if len(pairs) > 2:
                first = np.asarray([pair[0] for pair in pairs])
                second = np.asarray([pair[1] for pair in pairs])
                if first.std() > 0 and second.std() > 0:
                    reproducibility[variant] = float(np.corrcoef(first, second)[0, 1])
                else:
                    reproducibility[variant] = None
            print(
                f"[study {config.seed}] verification variant {variant}: "
                f"mean success {result['mean_success']:.3f}",
                flush=True,
            )
    with installed_variant(
        env,
        layers=layers,
        rank=config.rank,
        state_path=state_paths[know.TRANSFER_VARIANT_ID],
        variant_id=know.TRANSFER_VARIANT_ID,
    ):
        transfer_check = var.evaluate_facts(
            env,
            transfer_facts,
            template=var.HELD_OUT_TEMPLATE,
            draws=max(4, config.eval_draws // 2),
            seed=config.seed + 55,
            partition="verification_transfer",
        )
    verification = var.verify_knowledge(
        success_by_variant,
        known_facts=known_by_variant,
        train_template_success=train_template_success,
        format_stats=format_stats,
        reproducibility=reproducibility,
    )
    verification.details["transfer_variant_mean_success"] = transfer_check["mean_success"]
    write_json(out_dir / "evaluation" / "knowledge_verification.json", verification.to_json())
    print(
        f"[study {config.seed}] verification passed={verification.passed} "
        f"mean_spread={verification.details['mean_spread']:.3f}",
        flush=True,
    )
    if not verification.passed:
        return {
            "seed": config.seed,
            "stage": "verification_failed",
            "verification": verification.to_json(),
            "variant_training": training_metrics,
        }

    # --- reporter states ---------------------------------------------------------
    train_states = collect_all_variants(
        env,
        layers=layers,
        rank=config.rank,
        variant_states={variant: state_paths[variant] for variant in know.VARIANT_IDS},
        facts=splits["train"],
        template=var.HELD_OUT_TEMPLATE,
        draws=config.train_draws,
        seed=config.seed,
        partition="reporter_train",
    )
    dev_states = collect_all_variants(
        env,
        layers=layers,
        rank=config.rank,
        variant_states={variant: state_paths[variant] for variant in know.VARIANT_IDS},
        facts=splits["dev"],
        template=var.HELD_OUT_TEMPLATE,
        draws=config.train_draws,
        seed=config.seed + 1,
        partition="reporter_dev",
    )
    test_states = collect_all_variants(
        env,
        layers=layers,
        rank=config.rank,
        variant_states={variant: state_paths[variant] for variant in know.VARIANT_IDS},
        facts=splits["test"],
        template=var.HELD_OUT_TEMPLATE,
        draws=config.eval_draws,
        seed=config.seed + 2,
        partition="reporter_test",
    )
    transfer_states = collect_all_variants(
        env,
        layers=layers,
        rank=config.rank,
        variant_states={know.TRANSFER_VARIANT_ID: state_paths[know.TRANSFER_VARIANT_ID]},
        facts=transfer_facts,
        template=var.HELD_OUT_TEMPLATE,
        draws=config.eval_draws,
        seed=config.seed + 3,
        partition="transfer_unseen_variant",
    )

    # --- reporter training -------------------------------------------------------
    selection = select_lambda(
        train_states,
        dev_states,
        lambdas=config.lambda_grid,
        steps=config.reporter_steps,
        seed=config.seed,
    )
    t1 = selection["heads"]["0"]
    t2 = selection["heads"][str(selection["best_lambda"])]
    print(
        f"[study {config.seed}] lambda={selection['best_lambda']} dev scores "
        f"{ {key: round(value['total'], 4) for key, value in selection['scores'].items()} }",
        flush=True,
    )

    def evaluate_split(states: StateData, label: str) -> dict:
        report_t1 = evaluate_reporter(t1, states, label=f"{label}_t1")
        report_t2 = evaluate_reporter(t2, states, label=f"{label}_t2")
        comparison = compare_reporters(t1, t2, states, seed=config.seed)
        # Out-of-fold report predictions for the second-stage fit: in-sample
        # predictions would make the augmented predictor overtrust the reporter.
        oof_train = out_of_fold_report_predictions(
            train_states,
            lam=float(selection["best_lambda"]),
            steps=config.reporter_steps,
            seed=config.seed,
        )
        added_t2 = baseline_and_added_value(
            train_states,
            dev_states,
            states,
            oof_train,
            reporter_predictions(t2, states),
            seed=config.seed,
        )
        return {
            "t1": {key: report_t1[key] for key in ("brier", "brier_vs_success_mean", "contrast_error", "ece", "calibration", "mean_report", "mean_success")},
            "t2": {key: report_t2[key] for key in ("brier", "brier_vs_success_mean", "contrast_error", "ece", "calibration", "mean_report", "mean_success")},
            "comparison": comparison,
            "added_value_t2": added_t2["added_value"],
            "strongest_baseline": added_t2["strongest_baseline"],
        }

    test_eval = evaluate_split(test_states, "test_seen_variants_unseen_subjects")
    transfer_eval = evaluate_split(transfer_states, "transfer_unseen_variant")
    print(
        f"[study {config.seed}] test brier t1={test_eval['t1']['brier']:.4f} "
        f"t2={test_eval['t2']['brier']:.4f} added={test_eval['added_value_t2']['added']:.4f}",
        flush=True,
    )
    print(
        f"[study {config.seed}] transfer brier t1={transfer_eval['t1']['brier']:.4f} "
        f"t2={transfer_eval['t2']['brier']:.4f} added={transfer_eval['added_value_t2']['added']:.4f}",
        flush=True,
    )
    result = {
        "seed": config.seed,
        "stage": "complete",
        "layers": list(layers),
        "variant_training": training_metrics,
        "verification": verification.to_json(),
        "lambda_selection": {
            "best_lambda": selection["best_lambda"],
            "scores": selection["scores"],
        },
        "test": test_eval,
        "transfer_unseen_variant": transfer_eval,
        "states": {
            "train": _states_summary(train_states),
            "dev": _states_summary(dev_states),
            "test": _states_summary(test_states),
            "transfer": _states_summary(transfer_states),
        },
    }
    write_json(out_dir / "evaluation" / "knowledge_result.json", result)
    return result


def _states_summary(data: StateData) -> dict:
    return {
        "n_states": int(data.n_states),
        "n_questions": len({fact_id for fact_id in data.fact_ids}),
        "mean_success": float(data.success.mean()),
        "by_variant": {
            str(variant): float(
                data.success[np.asarray(data.variant_ids) == variant].mean()
            )
            for variant in sorted(set(data.variant_ids))
        },
    }
