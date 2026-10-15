"""Brier, contrast errors, calibration, transfer, and clustered intervals.

Primary quantities (plan section 11.2).  For an emitted report ``c`` and
independent evaluation outcomes ``Y``:

    BS = sum_{q,z} w_{qz} (1/K) sum_k (c_{qz} - Y_{qzk})^2
    Delta BS_added = BS(b) - BS(b + c)

where ``b + c`` means a predictor refitted with confidence as an additional
input.  Weights sum to one.  Within-question contrast error is

    CE_Delta = E[( (c_qa - c_qb) - (p_qa - p_qb) )^2].

The evaluation outcome bank is noisy; comparisons across reporter arms use
the same bank, and the subject-clustered bootstrap resamples whole subjects
with all their conditions, pairs, and draws (plan section 11.3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .protocol import CONDITION_WEIGHTS, PROTOCOL


def _as_matrix(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    return values


def _as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _normalized_weights(n: int, weights) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / max(n, 1))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != rows {n}")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return weights / total


def brier(scores, outcomes, weights=None) -> float:
    """Weighted Brier score.  ``outcomes`` may be ``(n,)`` or ``(n, K)``."""

    scores = _as_vector(scores)
    outcomes = _as_matrix(outcomes)
    if scores.shape[0] != outcomes.shape[0]:
        raise ValueError("scores and outcomes must have the same number of rows")
    error = (scores[:, None] - outcomes) ** 2
    per_row = error.mean(axis=1)
    weights = _normalized_weights(len(per_row), weights)
    return float((per_row * weights).sum())


def brier_parts(scores, outcomes) -> np.ndarray:
    """Per-row Brier values (mean over draws) for paired comparisons."""

    scores = _as_vector(scores)
    outcomes = _as_matrix(outcomes)
    return ((scores[:, None] - outcomes) ** 2).mean(axis=1)


def added_value(baseline_scores, augmented_scores, outcomes, weights=None) -> dict:
    """``BS(b) - BS(b + c)`` with both predictions fitted the same way.

    Positive values mean the report added predictive value.
    """

    base = brier(baseline_scores, outcomes, weights=weights)
    augmented = brier(augmented_scores, outcomes, weights=weights)
    return {"baseline": base, "augmented": augmented, "added": base - augmented}


def contrast_error(
    reports_a, reports_b, p_a, p_b, weights=None
) -> float:
    """Squared error of within-question report contrasts against success contrasts."""

    difference = (_as_vector(reports_a) - _as_vector(reports_b)) - (
        _as_vector(p_a) - _as_vector(p_b)
    )
    weights = _normalized_weights(len(difference), weights)
    return float((difference**2 * weights).sum())


def squared_probability_error(reports, probabilities, weights=None) -> float:
    """Squared distance from the true success probability, not calibration alone."""

    difference = _as_vector(reports) - _as_vector(probabilities)
    weights = _normalized_weights(len(difference), weights)
    return float((difference**2 * weights).sum())


def mixture_weights(slots: list[str], mapping: dict | None = None) -> np.ndarray:
    mapping = mapping or CONDITION_WEIGHTS
    return np.asarray([mapping[slot] for slot in slots], dtype=np.float64)


# ---------------------------------------------------------------------------
# Calibration and secondary metrics
# ---------------------------------------------------------------------------
def reliability_table(scores, outcomes, *, bins: int = 10) -> list[dict]:
    scores = _as_vector(scores)
    outcomes = _as_matrix(outcomes).mean(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (scores >= low) & (scores < high if index < bins - 1 else scores <= high)
        if not mask.any():
            table.append({"low": float(low), "high": float(high), "n": 0})
            continue
        table.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "mean_outcome": float(outcomes[mask].mean()),
            }
        )
    return table


def expected_calibration_error(scores, outcomes, *, bins: int = 10) -> float:
    scores = _as_vector(scores)
    outcomes = _as_matrix(outcomes).mean(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (scores >= low) & (scores < high if index < bins - 1 else scores <= high)
        if not mask.any():
            continue
        gap = abs(scores[mask].mean() - outcomes[mask].mean())
        total += gap * mask.mean()
    return float(total)


def auroc(scores, outcomes) -> float:
    """Rank AUROC against per-row outcome means (ties handled by ranking)."""

    scores = _as_vector(scores)
    labels = _as_matrix(outcomes).mean(axis=1)
    positive = labels > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    n_pos = positive.sum()
    n_neg = negative.sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_slope_intercept(scores, outcomes) -> dict:
    """Logistic calibration of the score's logit against outcome draws."""

    from sklearn.linear_model import LogisticRegression

    scores = np.clip(_as_vector(scores), 1e-6, 1 - 1e-6)
    outcomes = _as_matrix(outcomes)
    logits = np.log(scores / (1 - scores))[:, None]
    expanded = np.repeat(logits, outcomes.shape[1], axis=0)
    labels = outcomes.reshape(-1)
    model = LogisticRegression(max_iter=2_000, C=1e6)
    model.fit(expanded, labels)
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def selective_risk(
    scores,
    outcomes,
    *,
    coverages: tuple[float, ...] = (0.25, 0.50, 0.75),
    target_error_rate: float = 0.10,
) -> dict:
    """Risk at fixed coverage and coverage at a fixed error budget."""

    scores = _as_vector(scores)
    errors = (_as_matrix(outcomes).mean(axis=1) < 0.5).astype(np.float64)
    order = np.argsort(-scores)
    results: dict = {"at_coverage": {}}
    for coverage in coverages:
        take = max(1, int(round(coverage * len(scores))))
        kept = order[:take]
        results["at_coverage"][str(coverage)] = {
            "coverage": take / len(scores),
            "risk": float(errors[kept].mean()),
        }
    # Largest coverage whose risk stays under the target error rate.
    best = 0.0
    for take in range(1, len(scores) + 1):
        kept = order[:take]
        if errors[kept].mean() <= target_error_rate:
            best = take / len(scores)
    results["coverage_at_target_error"] = {
        "target_error": target_error_rate,
        "coverage": best,
    }
    return results


def sign_agreement(delta_report, delta_true, *, min_effect: float | None = None) -> dict:
    """Sign agreement between report changes and success changes.

    Report the denominator and, when given, restrict to pairs whose measured
    success change exceeds ``min_effect`` so tiny noisy differences do not
    dominate.
    """

    delta_report = _as_vector(delta_report)
    delta_true = _as_vector(delta_true)
    mask = np.ones(len(delta_report), dtype=bool)
    if min_effect is not None:
        mask = np.abs(delta_true) >= min_effect
    if not mask.any():
        return {"n": 0, "agreement": None, "min_effect": min_effect}
    same = np.sign(delta_report[mask]) == np.sign(delta_true[mask])
    return {
        "n": int(mask.sum()),
        "agreement": float(same.mean()),
        "min_effect": min_effect,
    }


# ---------------------------------------------------------------------------
# Subject-clustered bootstrap
# ---------------------------------------------------------------------------
@dataclass
class ClusterItem:
    """One row with its subject cluster key and arbitrary payload."""

    subject_id: str
    payload: dict


def _cluster_indices(items: list[ClusterItem]) -> dict[str, list[int]]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        clusters[item.subject_id].append(index)
    return dict(clusters)


def cluster_bootstrap(
    items: list[ClusterItem],
    metric_fn,
    *,
    n_samples: int = PROTOCOL.gates.bootstrap_samples,
    seed: int = PROTOCOL.gates.bootstrap_seed,
    level: float = PROTOCOL.gates.confidence_level,
) -> dict:
    """Resample subjects with replacement; keep every row of a resampled subject."""

    clusters = _cluster_indices(items)
    cluster_keys = list(clusters)
    generator = np.random.default_rng(seed)
    estimate = float(metric_fn(list(range(len(items))), items))
    samples = np.empty(n_samples, dtype=np.float64)
    for sample in range(n_samples):
        chosen = generator.integers(0, len(cluster_keys), size=len(cluster_keys))
        index: list[int] = []
        for position in chosen:
            index.extend(clusters[cluster_keys[position]])
        samples[sample] = float(metric_fn(index, items))
    alpha = (1 - level) / 2
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1 - alpha)),
        "n_samples": n_samples,
        "n_clusters": len(cluster_keys),
        "n_items": len(items),
    }


def paired_cluster_bootstrap(
    items: list[ClusterItem],
    metric_fn,
    *,
    n_samples: int = PROTOCOL.gates.bootstrap_samples,
    seed: int = PROTOCOL.gates.bootstrap_seed,
    level: float = PROTOCOL.gates.confidence_level,
) -> dict:
    """Bootstrap interval for a paired contrast computed by ``metric_fn``."""

    return cluster_bootstrap(
        items, metric_fn, n_samples=n_samples, seed=seed, level=level
    )


def gate_decision(
    effect: float,
    ci_low: float,
    threshold: float,
    *,
    direction: str = "greater",
) -> dict:
    """Apply one prespecified gate: effect threshold plus interval above zero."""

    if direction == "greater":
        passed = effect >= threshold and ci_low > 0.0
    elif direction == "less":
        passed = effect <= threshold and ci_low < 0.0
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return {
        "effect": effect,
        "ci_low": ci_low,
        "threshold": threshold,
        "direction": direction,
        "passed": bool(passed),
    }


def guardrail_upper_bound(values: np.ndarray, *, level: float = 0.95, seed: int = 0) -> dict:
    """One-sided upper confidence bound for a per-row degradation metric."""

    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.array(
        [values[generator.integers(0, len(values), size=len(values))].mean() for _ in range(2_000)]
    )
    return {
        "mean": float(values.mean()),
        "upper": float(np.quantile(samples, level)),
        "level": level,
    }
