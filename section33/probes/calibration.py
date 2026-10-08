"""Probability calibration and risk-coverage utilities.

The calibration map is fitted **only** on the calibration split, after method
selection.  Identity, Platt scaling, and isotonic regression compete through
grouped internal cross-validation on that split (three folds).  The chosen map
is refitted on the full calibration split and serialised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from ..config import PROTOCOL


@dataclass
class Calibrator:
    method: str
    model: object | None = None

    def apply(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "identity":
            return np.clip(scores, 0.0, 1.0)
        if self.method == "platt":
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(scores), 0.0, 1.0)
        raise ValueError(self.method)


def _fit_method(method: str, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
    if method == "identity":
        return Calibrator("identity")
    if method == "platt":
        model = LogisticRegression(C=1e6, max_iter=2000).fit(scores.reshape(-1, 1), labels)
        return Calibrator("platt", model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        return Calibrator("isotonic", model)
    raise ValueError(method)


def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    methods: tuple[str, ...] = PROTOCOL.calibration_methods,
    seed: int = 0,
) -> tuple[Calibrator, dict]:
    """Choose and refit the calibration map on the calibration split only."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups)
    folds = StratifiedGroupKFold(n_splits=PROTOCOL.inner_folds, shuffle=True, random_state=seed)
    records = []
    for method in methods:
        losses = []
        for train_index, valid_index in folds.split(scores.reshape(-1, 1), labels, groups):
            calibrator = _fit_method(method, scores[train_index], labels[train_index])
            prediction = calibrator.apply(scores[valid_index])
            losses.append(brier_score_loss(labels[valid_index], np.clip(prediction, 0, 1)))
        records.append({"method": method, "mean_brier": float(np.mean(losses))})
    best = min(records, key=lambda record: record["mean_brier"])
    calibrator = _fit_method(best["method"], scores, labels)
    return calibrator, {"selected": best, "candidates": records}


def reliability_bins(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray | None = None) -> dict:
    """Fixed-bin reliability with counts, at the frozen ten-bin edges."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    counts, bin_edges = np.histogram(scores, bins=edges)
    rows = []
    for index in range(len(bin_edges) - 1):
        mask = (scores >= bin_edges[index]) & (scores < bin_edges[index + 1])
        if index == len(bin_edges) - 2:
            mask = mask | (scores == bin_edges[index + 1])
        n = int(mask.sum())
        rows.append(
            {
                "bin_low": float(bin_edges[index]),
                "bin_high": float(bin_edges[index + 1]),
                "n": n,
                "mean_score": float(scores[mask].mean()) if n else None,
                "mean_label": float(labels[mask].mean()) if n else None,
            }
        )
    return {"edges": edges.tolist(), "counts": counts.tolist(), "bins": rows}


def corrected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> float:
    """Variance-corrected squared bin-calibration error.

    For residuals ``r = y - p`` in bin ``b`` the contribution is
    ``mean(r)^2 - var(r) / n_b``.  With subject clusters the variance term is
    replaced by a cluster-aware estimate of the variance of the bin mean.
    Negative contributions are kept (the caller compares methods directly).
    """

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(scores) == 0:
        return 0.0
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    total = 0.0
    for index in range(len(edges) - 1):
        mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if index == len(edges) - 2:
            mask = mask | (scores == edges[index + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        residual = labels[mask] - scores[mask]
        mean_r = residual.mean()
        if groups is None:
            variance_of_mean = residual.var(ddof=1) / n if n > 1 else 0.0
        else:
            # Design-aware variance of the observation-weighted bin mean.  The
            # cluster totals are centered by the cluster's expected contribution
            # (n_c * mean_r), so clusters of different sizes are handled
            # correctly.  With one observation per cluster this reduces to the
            # unweighted variance of the cluster means.
            cluster_ids = np.asarray(groups)[mask]
            unique = np.unique(cluster_ids)
            n_clusters = len(unique)
            if n_clusters > 1:
                totals = np.array([residual[cluster_ids == c].sum() for c in unique])
                sizes = np.array([float((cluster_ids == c).sum()) for c in unique])
                centered = totals - sizes * mean_r
                variance_of_mean = float(
                    np.sum(centered**2) / (n**2) * (n_clusters / (n_clusters - 1))
                )
            else:
                variance_of_mean = 0.0
        # Negative contributions are kept: the plan requires comparing methods
        # without clipping (section 9.2).
        total += (n / len(scores)) * (mean_r**2 - variance_of_mean)
    return float(total)


def risk_coverage(
    scores: np.ndarray,
    labels: np.ndarray,
    coverages: tuple[float, ...] = PROTOCOL.coverage_levels,
) -> list[dict]:
    """Error rate among the retained answers at fixed coverage levels."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(-scores)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(scores))))
        selected = order[:keep]
        rows.append(
            {
                "coverage": coverage,
                "kept": int(keep),
                "error_rate": float(1.0 - labels[selected].mean()),
            }
        )
    return rows


def threshold_for_error(
    scores: np.ndarray,
    labels: np.ndarray,
    target_error: float = PROTOCOL.target_error_rate,
) -> dict:
    """Pick the smallest threshold whose calibration error is at most target."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(-scores)
    thresholds = np.unique(scores)
    best = {"threshold": None, "coverage": 0.0, "error": None}
    for threshold in thresholds:
        mask = scores >= threshold
        if mask.sum() == 0:
            continue
        error = float(1.0 - labels[mask].mean())
        if error <= target_error:
            best = {
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "error": error,
            }
    return best
