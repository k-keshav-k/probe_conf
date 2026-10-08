"""Paired evaluation and frozen decision rules (plan sections 9 and 10).

Primary estimands (Track A, fresh final test):

    Delta_AUC   = AUROC(B+M) - AUROC(B)
    Delta_Brier = mean((B - Y)^2) - mean((B+M - Y)^2)

Positive values are improvements.  Intervals are paired subject bootstraps with
stratum-aware resampling.  Holm correction applies to separately claimed
secondary comparisons.  ``decide_primary`` applies the registered yes/no rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import PROTOCOL


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def clip_probabilities(scores: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), epsilon, 1 - epsilon)


def log_loss(labels: np.ndarray, scores: np.ndarray, epsilon: float = 1e-6) -> float:
    p = clip_probabilities(scores, epsilon)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Stratum-aware paired subject bootstrap
# ---------------------------------------------------------------------------
def _subject_table(df, score_a: str, score_b: str, label: str, subject: str, strata: list[str]):
    grouped = df.groupby(subject)
    records = []
    for subject_id, block in grouped:
        strata_key = tuple(block[strata_column].iloc[0] for strata_column in strata) if strata else ()
        records.append(
            {
                "subject": subject_id,
                "strata": strata_key,
                "labels": block[label].to_numpy(),
                "a": block[score_a].to_numpy(),
                "b": block[score_b].to_numpy(),
            }
        )
    return records


def paired_subject_bootstrap(
    df,
    score_a: str,
    score_b: str,
    label: str,
    subject: str = "subject_id",
    strata: tuple[str, ...] = ("relation", "stratum"),
    n_boot: int = PROTOCOL.bootstrap_samples,
    seed: int = PROTOCOL.bootstrap_seed,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap over subjects within strata.

    Returns intervals for ``Delta_AUC = A - B`` and
    ``Delta_Brier = Brier(B) - Brier(A)`` (positive = A is better).
    """

    records = _subject_table(df, score_a, score_b, label, subject, list(strata))
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["strata"], []).append(index)

    deltas_auc, deltas_brier, auc_a, auc_b = [], [], [], []
    for _ in range(n_boot):
        sampled = []
        for key, indices in groups.items():
            drawn = rng.choice(indices, size=len(indices), replace=True)
            sampled.extend(drawn)
        labels = np.concatenate([records[i]["labels"] for i in sampled])
        a = np.concatenate([records[i]["a"] for i in sampled])
        b = np.concatenate([records[i]["b"] for i in sampled])
        if len(np.unique(labels)) < 2:
            continue
        auc_a.append(roc_auc_score(labels, a))
        auc_b.append(roc_auc_score(labels, b))
        deltas_auc.append(auc_a[-1] - auc_b[-1])
        deltas_brier.append(
            brier_score_loss(labels, np.clip(b, 0, 1)) - brier_score_loss(labels, np.clip(a, 0, 1))
        )

    def interval(values):
        array = np.asarray(values)
        alpha = 1.0 - confidence
        return (
            float(np.percentile(array, 100 * alpha / 2)),
            float(np.percentile(array, 100 * (1 - alpha / 2))),
        )

    labels_all = df[label].to_numpy()
    a_all = df[score_a].to_numpy()
    b_all = df[score_b].to_numpy()
    return {
        "n_subjects": len(records),
        "n_observations": int(len(df)),
        "auroc_a": auroc(labels_all, a_all),
        "auroc_b": auroc(labels_all, b_all),
        "delta_auc": float(np.mean(deltas_auc)),
        "delta_auc_interval": interval(deltas_auc),
        "delta_brier": float(np.mean(deltas_brier)),
        "delta_brier_interval": interval(deltas_brier),
        "delta_auc_samples": [float(v) for v in deltas_auc],
        "delta_brier_samples": [float(v) for v in deltas_brier],
        "bootstrap_seed": seed,
        "n_bootstrap": len(deltas_auc),
        "confidence": confidence,
    }


def bootstrap_p_value(deltas: list[float] | np.ndarray) -> float:
    """Two-sided bootstrap p-value for a paired improvement.

    The null value is zero.  The p-value is twice the smaller tail probability,
    capped at one.  Multiplicity across comparisons is handled by the caller
    with ``holm``.
    """

    values = np.asarray(deltas, dtype=float)
    if len(values) == 0:
        return float("nan")
    left = float(np.mean(values <= 0))
    right = float(np.mean(values >= 0))
    return float(min(1.0, 2.0 * min(left, right)))


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------
def holm(p_values: list[float], family_size: int | None = None) -> list[float]:
    """Holm step-down adjusted p-values (familywise 0.05 by default)."""

    m = family_size or len(p_values)
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Frozen decision rules (plan section 10.3)
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    verdict: str
    reason: str
    details: dict

    def to_row(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason, **self.details}


def decide_primary(
    result: dict,
    calibration_error_increase: tuple[float, float] | None = None,
) -> Decision:
    """Apply the primary Track A rules.

    ``result`` comes from ``paired_subject_bootstrap``.  ``calibration_error_increase``
    is the interval of the increase in the corrected calibration error.
    """

    lo_auc, hi_auc = result["delta_auc_interval"]
    lo_brier, hi_brier = result["delta_brier_interval"]
    margins = PROTOCOL.margins
    details = {
        "delta_auc": result["delta_auc"],
        "delta_auc_interval": [lo_auc, hi_auc],
        "delta_brier": result["delta_brier"],
        "delta_brier_interval": [lo_brier, hi_brier],
    }
    calibration_ok = True
    if calibration_error_increase is not None:
        details["calibration_error_increase"] = list(calibration_error_increase)
        calibration_ok = calibration_error_increase[1] < margins.calibration_epsilon
    details["calibration_ok"] = calibration_ok

    if lo_auc > margins.main_auroc and lo_brier > margins.main_brier and calibration_ok:
        return Decision(
            "useful_improvement_supported",
            "both lower bounds exceed the registered practical margins and the calibration guardrail passes",
            details,
        )
    if hi_auc < margins.main_auroc and hi_brier < margins.main_brier:
        return Decision(
            "useful_improvement_ruled_out",
            "both upper bounds stay below the registered practical margins",
            details,
        )
    if hi_auc < 0 or hi_brier < 0:
        return Decision(
            "harm_supported",
            "an improvement interval lies wholly below zero",
            details,
        )
    return Decision(
        "inconclusive",
        "intervals cross the practical margins or the calibration guardrail is unresolved",
        details,
    )


def decide_within_question(
    delta_interval: tuple[float, float],
    brier_increase_interval: tuple[float, float],
    margin: float | None = None,
) -> Decision:
    margin = margin or PROTOCOL.margins.within_question_auroc
    lo, hi = delta_interval
    b_lo, b_hi = brier_increase_interval
    details = {
        "within_question_delta_interval": [lo, hi],
        "all_answer_brier_increase_interval": [b_lo, b_hi],
    }
    if lo > margin and b_hi < 0.002:
        return Decision(
            "within_question_gain_supported",
            "adjusted lower bound exceeds the margin and the all-answer Brier guardrail passes",
            details,
        )
    if hi < margin:
        return Decision(
            "within_question_gain_ruled_out",
            "adjusted upper bound stays below the registered margin",
            details,
        )
    return Decision("within_question_inconclusive", "interval crosses the margin", details)


def decide_equivalence(interval: tuple[float, float], margin: float | None = None) -> Decision:
    margin = margin or PROTOCOL.margins.equivalence_auroc
    lo, hi = interval
    if lo >= -margin and hi <= margin:
        return Decision("practical_equivalence", "interval inside the equivalence margin", {"interval": [lo, hi]})
    if lo > 0:
        return Decision("mlp_superior", "interval above zero", {"interval": [lo, hi]})
    if hi < 0:
        return Decision("hidden_superior", "interval below zero", {"interval": [lo, hi]})
    return Decision("equivalence_not_established", "interval crosses zero", {"interval": [lo, hi]})


def calibration_guardrail_interval(
    df,
    score_a: str,
    score_b: str,
    label: str,
    subject: str = "subject_id",
    strata: tuple[str, ...] = ("relation", "stratum"),
    n_boot: int = PROTOCOL.bootstrap_samples,
    seed: int = PROTOCOL.bootstrap_seed,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Paired interval of ``error(score_a) - error(score_b)``.

    Positive values mean that ``score_a`` (the first argument) is worse
    calibrated.  ``cmd_evaluate`` passes the augmented score first, so the
    interval is the increase that the primary guardrail checks.

    Each bootstrap copy receives a fresh cluster id, so repeated draws of one
    subject are treated as the independent copies they are rather than being
    collapsed back into one cluster.
    """

    from ..probes.calibration import corrected_calibration_error

    records = _subject_table(df, score_a, score_b, label, subject, list(strata))
    rng = np.random.default_rng(seed + 1)
    groups: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["strata"], []).append(index)
    values = []
    for _ in range(n_boot):
        sampled = []
        for key, indices in groups.items():
            sampled.extend(rng.choice(indices, size=len(indices), replace=True))
        labels = np.concatenate([records[i]["labels"] for i in sampled])
        a = np.concatenate([records[i]["a"] for i in sampled])
        b = np.concatenate([records[i]["b"] for i in sampled])
        subjects = np.concatenate(
            [np.full(len(records[i]["labels"]), copy) for copy, i in enumerate(sampled)]
        )
        error_a = corrected_calibration_error(a, labels, groups=subjects)
        error_b = corrected_calibration_error(b, labels, groups=subjects)
        values.append(error_a - error_b)
    alpha = 1.0 - confidence
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def _pairwise_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def paired_within_question_bootstrap(
    df,
    score_a: str,
    score_b: str,
    label: str = "label",
    question: str = "question_id",
    subject: str = "subject_id",
    strata: tuple[str, ...] = ("relation", "stratum"),
    n_boot: int = PROTOCOL.bootstrap_samples,
    seed: int = PROTOCOL.bootstrap_seed,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap over subjects of the mean per-question AUROC difference.

    Only mixed-outcome questions enter.  Every eligible question contributes
    one value, so the estimate is the registered within-question estimand.
    """

    per_subject: dict = {}
    eligible = 0
    total_questions = df[question].nunique()
    for question_id, block in df.groupby(question):
        y = block[label].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        a = block[score_a].to_numpy()
        b = block[score_b].to_numpy()
        delta = _pairwise_auroc(a[y == 1], a[y == 0]) - _pairwise_auroc(b[y == 1], b[y == 0])
        subject_id = block[subject].iloc[0]
        strata_key = tuple(block[column].iloc[0] for column in strata)
        record = per_subject.setdefault(subject_id, {"strata": strata_key, "deltas": []})
        record["deltas"].append(delta)
        eligible += 1
    groups: dict[tuple, list] = {}
    for subject_id, record in per_subject.items():
        groups.setdefault(record["strata"], []).append(subject_id)
    rng = np.random.default_rng(seed + 2)
    values = []
    for _ in range(n_boot):
        sampled = []
        for key, subject_ids in groups.items():
            sampled.extend(rng.choice(subject_ids, size=len(subject_ids), replace=True))
        deltas = np.concatenate([per_subject[subject_id]["deltas"] for subject_id in sampled])
        values.append(float(deltas.mean()))
    alpha = 1.0 - confidence
    return {
        "within_question_delta": float(np.mean(values)),
        "within_question_delta_interval": [
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))),
        ],
        "eligible_questions": int(eligible),
        "n_questions": int(total_questions),
        "confidence": confidence,
    }
