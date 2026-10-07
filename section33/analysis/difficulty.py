"""Difficulty control for the Section 3.3 experiment.

Plan section 7:

- train-only popularity and difficulty strata, frozen on test data;
- within-question AUROC for Track B (ties count as one half);
- cross-fitted difficulty predictors for binary labels and binomial counts;
- residuals computed only after both sides share the same target;
- balanced-population reporting by relation and stratum.

Nothing here uses test labels to choose bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from ..config import POPULARITY_BINS


# ---------------------------------------------------------------------------
# Strata fitted on training data only
# ---------------------------------------------------------------------------
@dataclass
class Strata:
    edges: list[float]
    column: str

    def apply(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        bucketed = np.searchsorted(np.asarray(self.edges), values, side="right") - 1
        return np.clip(bucketed, 0, len(self.edges) - 2)


def fit_popularity_strata(train_values: np.ndarray, column: str = "log_popularity") -> Strata:
    values = np.asarray(train_values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < POPULARITY_BINS:
        raise ValueError("not enough popularity values for bins")
    edges = np.quantile(values, np.linspace(0, 1, POPULARITY_BINS + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return Strata(edges=list(map(float, edges)), column=column)


def fit_difficulty_score_strata(train_scores: np.ndarray, n_bins: int = POPULARITY_BINS) -> Strata:
    return fit_popularity_strata(train_scores, column="difficulty_score")


# ---------------------------------------------------------------------------
# Within-question AUROC (all eight answers of one question)
# ---------------------------------------------------------------------------
def within_question_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray,
) -> dict:
    """Mean within-question AUROC over mixed-outcome questions.

    Ties count as one half.  Each eligible question contributes one value, so
    prolific subjects and questions with many samples cannot dominate.
    Candidate multiplicity is handled by the caller: expand rows by
    ``multiplicity`` before calling (``datasets.expand_by_multiplicity``).
    """

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    question_ids = np.asarray(question_ids)
    values = []
    mixed = 0
    for question in np.unique(question_ids):
        mask = question_ids == question
        y = labels[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2:
            continue
        mixed += 1
        pos = s[y == 1]
        neg = s[y == 0]
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        values.append(float(wins / (len(pos) * len(neg))))
    return {
        "within_question_auroc": float(np.mean(values)) if values else None,
        "eligible_questions": int(mixed),
        "n_questions": int(len(np.unique(question_ids))),
    }


# ---------------------------------------------------------------------------
# Cross-fitted difficulty predictors
# ---------------------------------------------------------------------------
def cross_fitted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions of a binary difficulty model."""

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups)
    folds = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    predictions = np.full(len(labels), np.nan)
    for train_index, test_index in folds.split(features, labels, groups):
        model = LogisticRegression(max_iter=3000, C=0.1).fit(features[train_index], labels[train_index])
        predictions[test_index] = model.predict_proba(features[test_index])[:, 1]
    return predictions


def fit_binomial_logistic(
    features: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    l2: float = 1e-3,
    max_iter: int = 100,
) -> np.ndarray:
    """IRLS fit of a logistic model on binomial counts.  Returns coefficients."""

    features = np.asarray(features, dtype=float)
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    design = np.hstack([np.ones((len(features), 1)), features])
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(totals * probability * (1 - probability), 1e-12, None)
        working = eta + (successes - totals * probability) / weight
        ridge = l2 * np.eye(design.shape[1])
        ridge[0, 0] = 0.0
        lhs = design.T @ (design * weight[:, None]) + ridge
        rhs = design.T @ (weight * working)
        beta_new = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_binomial(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((len(features), 1)), np.asarray(features, dtype=float)])
    return 1.0 / (1.0 + np.exp(-(design @ beta)))


def cross_fitted_binomial(
    features: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold binomial difficulty predictions (one per question)."""

    features = np.asarray(features, dtype=float)
    groups = np.asarray(groups)
    folds = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    predictions = np.full(len(features), np.nan)
    for train_index, test_index in folds.split(features, successes, groups):
        beta = fit_binomial_logistic(
            features[train_index], successes[train_index], totals[train_index]
        )
        predictions[test_index] = predict_binomial(features[test_index], beta)
    return predictions


def residual_correlation(residual: np.ndarray, score: np.ndarray) -> dict:
    """Spearman correlation of a residual with a score (secondary diagnostic)."""

    from scipy.stats import spearmanr

    mask = np.isfinite(residual) & np.isfinite(score)
    if mask.sum() < 3:
        return {"rho": None, "n": int(mask.sum())}
    rho = spearmanr(residual[mask], np.asarray(score)[mask]).statistic
    return {"rho": float(rho), "n": int(mask.sum())}


def masked_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))
