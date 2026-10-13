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

from .config import CONDITIONS, REPORT_CONDITIONS, ToyConfig
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
                }
            )
    return rows


def _features(rows: list[dict], n_relations: int) -> np.ndarray:
    features = []
    for row in rows:
        relation = np.zeros(n_relations)
        relation[row["relation"]] = 1.0
        condition = np.zeros(len(REPORT_CONDITIONS))
        condition[REPORT_CONDITIONS.index(row["condition"])] = 1.0
        features.append(
            np.concatenate(
                [relation, condition, [row["max_prob"], row["entropy"]]]
            )
        )
    return np.asarray(features, dtype=np.float64)


def _outcome_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["outcomes"] for row in rows], dtype=np.float64)


def _brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probabilities = probabilities[:, None]
    return float(np.mean((probabilities - outcomes) ** 2))


def reporting_metrics(
    train_rows: list[dict],
    test_rows: list[dict],
    n_relations: int,
) -> dict:
    """Brier endpoints with a metadata plus output-probability baseline."""

    from sklearn.linear_model import LogisticRegression

    train_x = _features(train_rows, n_relations)
    train_outcomes = _outcome_matrix(train_rows)
    train_y = train_outcomes.reshape(-1)
    train_x = np.repeat(train_x, train_outcomes.shape[1], axis=0)
    train_report = np.repeat(
        np.asarray([row["report"] for row in train_rows])[:, None], train_outcomes.shape[1], axis=0
    )
    test_x = _features(test_rows, n_relations)
    test_y = _outcome_matrix(test_rows)
    report = np.asarray([row["report"] for row in test_rows])[:, None]
    p_correct = np.asarray([row["p_correct"] for row in test_rows])[:, None]

    baseline = LogisticRegression(max_iter=2000, C=1.0)
    baseline.fit(train_x, train_y)
    baseline_prob = baseline.predict_proba(test_x)[:, 1][:, None]

    augmented = LogisticRegression(max_iter=2000, C=1.0)
    augmented.fit(np.hstack([train_x, train_report]), train_y)
    augmented_prob = augmented.predict_proba(np.hstack([test_x, report]))[:, 1][:, None]

    estimates = np.asarray(
        [
            _squared_error_estimate(row["report"], np.asarray(row["outcomes"], dtype=np.float64))
            for row in test_rows
        ]
    )

    same_question = _same_question_tracking(test_rows)

    return {
        "brier_report": _brier(np.asarray([row["report"] for row in test_rows]), test_y),
        "brier_baseline": _brier(baseline_prob, test_y),
        "brier_augmented": _brier(augmented_prob, test_y),
        "incremental_brier": _brier(baseline_prob, test_y) - _brier(augmented_prob, test_y),
        "brier_policy": _brier(p_correct, test_y),
        "squared_error_estimate": float(np.mean(estimates)),
        "same_question": same_question,
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
    lookup: dict[tuple[int, int, str], dict] = {}
    for row in test_rows:
        lookup[(row["subject_id"], row["relation"], row["condition"])] = row
    pairs = []
    for (subject, relation, condition), clean_row in lookup.items():
        if condition != "clean":
            continue
        for other in ("target_weaken", "irrelevant_weaken", "irrelevant_swap"):
            other_row = lookup.get((subject, relation, other))
            if other_row is not None:
                pairs.append(
                    (
                        clean_row["p_correct"] - other_row["p_correct"],
                        clean_row["report"] - other_row["report"],
                    )
                )
    if not pairs:
        return {"n": 0}
    delta_p = np.asarray([pair[0] for pair in pairs])
    delta_c = np.asarray([pair[1] for pair in pairs])
    sign_agreement = float(np.mean(np.sign(delta_p) == np.sign(delta_c)))
    correlation = float(np.corrcoef(delta_p, delta_c)[0, 1]) if np.std(delta_p) > 0 and np.std(delta_c) > 0 else float("nan")
    return {
        "n": len(pairs),
        "mean_delta_p": float(delta_p.mean()),
        "mean_delta_c": float(delta_c.mean()),
        "sign_agreement": sign_agreement,
        "correlation": correlation,
    }
