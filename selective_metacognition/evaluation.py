"""Evaluation for the toy experiments.

E1 reports functional selectivity: clean correctness, stability under
irrelevant changes, sensitivity to target weakening, response to valid swaps,
and the constant-output checks (proposal section 11.3).

E2 reports the confidence endpoints: the proper score, the unbiased
squared-probability-error estimate (proposal section 11.2), an incremental
Brier over a metadata and output-probability baseline, and same-question
condition tracking.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import CONDITIONS, PROSPECTIVE_TYPES, REPORT_CONDITIONS, ToyConfig
from .model import Selector
from .world import World

EPS = 1e-9


def all_pairs(subject_ids: torch.Tensor, n_relations: int) -> tuple[torch.Tensor, torch.Tensor]:
    subjects = subject_ids.repeat_interleave(n_relations)
    relations = torch.arange(n_relations).repeat(len(subject_ids))
    return subjects, relations


@torch.no_grad()
def condition_metrics(
    model: Selector,
    world: World,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    name: str,
    *,
    strength: float | None = None,
    seed: int = 0,
    assignments: torch.Tensor | None = None,
) -> dict:
    memory, correct = world.condition(
        subjects, relations, name, strength=strength, seed=seed, assignments=assignments
    )
    relations = relations.to(memory.device)
    policy = model.policy(memory, relations)
    p_correct = policy.gather(1, correct[:, None]).squeeze(1)
    top1 = policy.argmax(dim=1)
    entropy = -(policy * (policy + EPS).log()).sum(dim=1)
    return {
        "success_prob": float(p_correct.mean()),
        "top1": float((top1 == correct).float().mean()),
        "entropy": float(entropy.mean()),
        "p_correct": p_correct,
        "policy": policy,
        "top1_answers": top1,
        "correct": correct,
        "memory": memory,
    }


@torch.no_grad()
def functional_table(
    model: Selector,
    world: World,
    subject_ids: torch.Tensor,
    *,
    assignments: torch.Tensor | None = None,
    strength: float | None = None,
    seed_base: int = 0,
) -> tuple[dict, dict]:
    subjects, relations = all_pairs(subject_ids, world.config.n_relations)
    table: dict = {}
    per_item: dict = {}
    for index, name in enumerate(CONDITIONS):
        metrics = condition_metrics(
            model,
            world,
            subjects,
            relations,
            name,
            strength=strength,
            seed=seed_base + 17 * index,
            assignments=assignments,
        )
        table[name] = {
            "success_prob": metrics["success_prob"],
            "top1": metrics["top1"],
            "entropy": metrics["entropy"],
        }
        per_item[name] = metrics

    table["selectivity"] = (
        table["irrelevant_weaken"]["success_prob"] - table["target_weaken"]["success_prob"]
    )
    table["irrelevant_swap_stability"] = (
        table["clean"]["success_prob"] - table["irrelevant_swap"]["success_prob"]
    )

    top1 = per_item["clean"]["top1_answers"]
    counts = torch.bincount(top1, minlength=world.config.n_answers).float()
    share = counts / counts.sum().clamp_min(1.0)
    table["top1_share"] = float(share.max())
    table["marginal_entropy"] = float(-(share * (share + EPS).log()).sum())
    return table, per_item


@torch.no_grad()
def collect_report_rows(
    model: Selector,
    world: World,
    subject_ids: torch.Tensor,
    *,
    conditions: tuple[str, ...] = REPORT_CONDITIONS,
    assignments: torch.Tensor | None = None,
    outcomes: int = 16,
    seed: int = 0,
) -> list[dict]:
    """One row per (subject, relation, condition) with a fresh outcome sample."""

    subjects, relations = all_pairs(subject_ids, world.config.n_relations)
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict] = []
    for index, name in enumerate(conditions):
        memory, correct = world.condition(
            subjects, relations, name, seed=seed + 31 * index, assignments=assignments
        )
        relations = relations.to(memory.device)
        policy = model.policy(memory, relations)
        p_correct = policy.gather(1, correct[:, None]).squeeze(1)
        report = model.expected_report(memory, relations)
        top2 = policy.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        answers = torch.multinomial(
            policy.detach().cpu(), outcomes, replacement=True, generator=generator
        ).to(correct.device)
        outcomes_tensor = (answers == correct[:, None]).float()
        for row_index in range(len(subjects)):
            rows.append(
                {
                    "subject_id": int(subjects[row_index]),
                    "relation": int(relations[row_index]),
                    "condition": name,
                    "p_correct": float(p_correct[row_index]),
                    "report": float(report[row_index]),
                    "outcomes": outcomes_tensor[row_index].tolist(),
                    "entropy": float(
                        -(policy[row_index] * (policy[row_index] + EPS).log()).sum()
                    ),
                    "max_prob": float(policy[row_index].max()),
                    "margin": float(margin[row_index]),
                }
            )
    return rows


def _features(rows: list[dict], n_relations: int, include_condition: bool = False) -> np.ndarray:
    """Metadata and prospective output-probability features.

    ``include_condition`` adds the intervention identity.  The ordinary
    baseline must not receive it: the reporter never sees the condition, and
    ``target_swap_valid`` and ``false_memory`` can produce identical inputs
    with different ground truths.  The privileged version is reported
    separately.
    """

    features = []
    for row in rows:
        relation = np.zeros(n_relations)
        relation[row["relation"]] = 1.0
        parts = [relation, [row["max_prob"], row["entropy"], row["margin"]]]
        if include_condition and "condition" in row:
            condition = np.zeros(len(REPORT_CONDITIONS))
            condition[REPORT_CONDITIONS.index(row["condition"])] = 1.0
            parts.append(condition)
        if "intervention" in row:
            intervention = np.zeros(len(PROSPECTIVE_TYPES))
            intervention[PROSPECTIVE_TYPES.index(row["intervention"])] = 1.0
            parts.append(np.concatenate([intervention, [row["strength"]]]))
        features.append(np.concatenate(parts))
    return np.asarray(features, dtype=np.float64)


def _outcome_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["outcomes"] for row in rows], dtype=np.float64)


def _brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error with per-row probabilities broadcast over outcomes.

    Accepts a 1-D probability vector or a 2-D column.  The earlier version
    always added a column, which turned an already-columnar input into
    ``(N, 1, 1)`` and silently averaged over the wrong broadcast shape.
    """

    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    return float(np.mean((probabilities - outcomes) ** 2))


def reporting_metrics(
    train_rows: list[dict],
    test_rows: list[dict],
    n_relations: int,
) -> dict:
    """Brier endpoints with ordinary, privileged, constant, and recalibrated predictors.

    The ordinary baseline receives metadata and prospective output-probability
    features (max probability, entropy, top-2 margin) but **not** the
    intervention identity.  The privileged baseline additionally receives the
    condition one-hot; it is reported separately because the reporter never
    sees that label.  ``brier_report_recalibrated`` refits the report alone, to
    test whether its ranking carries information even if its scale does not.
    """

    from sklearn.linear_model import LogisticRegression

    train_outcomes = _outcome_matrix(train_rows)
    repeats = train_outcomes.shape[1]
    train_y = train_outcomes.reshape(-1)
    test_y = _outcome_matrix(test_rows)

    def expand(features: np.ndarray) -> np.ndarray:
        return np.repeat(features, repeats, axis=0)

    ordinary_x = expand(_features(train_rows, n_relations, include_condition=False))
    privileged_x = expand(_features(train_rows, n_relations, include_condition=True))
    test_ordinary = _features(test_rows, n_relations, include_condition=False)
    test_privileged = _features(test_rows, n_relations, include_condition=True)
    train_report = np.repeat(
        np.asarray([row["report"] for row in train_rows])[:, None], repeats, axis=0
    )
    report = np.asarray([row["report"] for row in test_rows])[:, None]
    p_correct = np.asarray([row["p_correct"] for row in test_rows])[:, None]

    def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        model = LogisticRegression(max_iter=2000, C=1.0).fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1][:, None]

    ordinary = fit_predict(ordinary_x, train_y, test_ordinary)
    ordinary_augmented = fit_predict(
        np.hstack([ordinary_x, train_report]), train_y, np.hstack([test_ordinary, report])
    )
    privileged = fit_predict(privileged_x, train_y, test_privileged)
    privileged_augmented = fit_predict(
        np.hstack([privileged_x, train_report]), train_y, np.hstack([test_privileged, report])
    )
    recalibrated = fit_predict(train_report, train_y, report)

    estimates = np.asarray(
        [
            _squared_error_estimate(row["report"], np.asarray(row["outcomes"], dtype=np.float64))
            for row in test_rows
        ]
    )
    report_flat = np.asarray([row["report"] for row in test_rows])

    return {
        "brier_report": _brier(report_flat, test_y),
        "brier_report_recalibrated": _brier(recalibrated, test_y),
        "brier_constant_half": _brier(np.full(len(test_rows), 0.5), test_y),
        "brier_constant_rate": _brier(np.full(len(test_rows), float(train_y.mean())), test_y),
        "brier_policy": _brier(p_correct, test_y),
        "brier_ordinary_baseline": _brier(ordinary, test_y),
        "brier_ordinary_augmented": _brier(ordinary_augmented, test_y),
        "incremental_brier": _brier(ordinary, test_y) - _brier(ordinary_augmented, test_y),
        "brier_privileged_baseline": _brier(privileged, test_y),
        "brier_privileged_augmented": _brier(privileged_augmented, test_y),
        "incremental_privileged": _brier(privileged, test_y) - _brier(privileged_augmented, test_y),
        "squared_probability_error": float(np.mean(estimates)),
        "same_question": _same_question_tracking(test_rows),
    }


def _squared_error_estimate(report: float, outcomes: np.ndarray) -> float:
    """Unbiased estimate of ``(c - p)**2`` with at least two draws (section 11.2)."""

    count = len(outcomes)
    if count < 2:
        return float("nan")
    total = outcomes.sum()
    cross = total * total - (outcomes * outcomes).sum()
    return float(report * report - 2.0 * report * outcomes.mean() + cross / (count * (count - 1)))


def _same_question_tracking(test_rows: list[dict]) -> dict:
    """Within-item report changes versus success changes, by condition pair."""

    if not test_rows or "condition" not in test_rows[0]:
        return {"n": 0}

    lookup: dict[tuple[int, int, str], dict] = {}
    for row in test_rows:
        lookup[(row["subject_id"], row["relation"], row["condition"])] = row

    result: dict = {"n": 0}
    all_pairs: list[tuple[float, float]] = []
    for condition in ("target_weaken", "irrelevant_weaken", "irrelevant_swap"):
        pairs: list[tuple[float, float]] = []
        for (subject, relation, name), clean_row in lookup.items():
            if name != "clean":
                continue
            other_row = lookup.get((subject, relation, condition))
            if other_row is None:
                continue
            pairs.append(
                (
                    clean_row["p_correct"] - other_row["p_correct"],
                    clean_row["report"] - other_row["report"],
                )
            )
        result[condition] = _pair_stats(pairs)
        all_pairs.extend(pairs)

    overall = _pair_stats(all_pairs)
    result["n"] = overall.get("n", 0)
    for key in ("mean_delta_p", "mean_delta_c", "sign_agreement", "correlation"):
        if key in overall:
            result[key] = overall[key]
    return result


def _pair_stats(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n": 0}
    delta_p = np.asarray([pair[0] for pair in pairs])
    delta_c = np.asarray([pair[1] for pair in pairs])
    sign_agreement = float(np.mean(np.sign(delta_p) == np.sign(delta_c)))
    correlation = (
        float(np.corrcoef(delta_p, delta_c)[0, 1])
        if np.std(delta_p) > 0 and np.std(delta_c) > 0
        else float("nan")
    )
    return {
        "n": len(pairs),
        "mean_delta_p": float(delta_p.mean()),
        "mean_delta_c": float(delta_c.mean()),
        "sign_agreement": sign_agreement,
        "correlation": correlation,
    }
