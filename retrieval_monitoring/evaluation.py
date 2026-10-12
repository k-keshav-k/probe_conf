"""Primary monitoring endpoint for the retrieval-monitoring plan.

The endpoint is the paired held-out Brier improvement from adding the report to
a prespecified baseline:

    Delta Brier = mean over subjects of [(b - y)^2 - (a - y)^2]

where ``b`` is the baseline predictor and ``a`` is the same predictor family
augmented with the report ``c``.  Draws, states, and relations are averaged
equally within each subject before subjects are averaged.

Two baselines are supported:

- ``text``: question text, relation, popularity, and length.
- ``diagnostic``: the text baseline plus output-uncertainty summaries (entropy
  and margin) and the perturbation norm.  This is the primary comparison.

The within-question shuffle test permutes report-to-state assignments inside
each question, refits both predictors on the shuffled data, and recomputes the
full statistic.  A positive observed value without shuffle evidence supports
only the narrower predictive result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 0


@dataclass
class EndpointResult:
    n_subjects: int
    n_observations: int
    baseline_brier: float
    augmented_brier: float
    improvement: float
    improvement_ci: tuple[float, float]
    bootstrap_mean: float
    baseline_auroc: float
    augmented_auroc: float
    shuffle_mean: float | None
    shuffle_p_value: float | None
    n_shuffle: int
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_subjects": self.n_subjects,
            "n_observations": self.n_observations,
            "baseline_brier": self.baseline_brier,
            "augmented_brier": self.augmented_brier,
            "improvement": self.improvement,
            "improvement_ci": list(self.improvement_ci),
            "bootstrap_mean": self.bootstrap_mean,
            "baseline_auroc": self.baseline_auroc,
            "augmented_auroc": self.augmented_auroc,
            "shuffle_mean": self.shuffle_mean,
            "shuffle_p_value": self.shuffle_p_value,
            "n_shuffle": self.n_shuffle,
            "provenance": self.provenance,
        }


def _fit_logistic(train: pd.DataFrame, columns: list[str]) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
    model.fit(train[columns].to_numpy(dtype=float), train["y"].to_numpy(dtype=int))
    return model


def _scores(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame[columns].to_numpy(dtype=float))[:, 1]


def paired_improvement(frame: pd.DataFrame, baseline: np.ndarray, augmented: np.ndarray) -> pd.Series:
    """Per-subject mean Brier improvement (equal subject weight)."""

    labels = frame["y"].to_numpy(dtype=float)
    improvement = (baseline - labels) ** 2 - (augmented - labels) ** 2
    return pd.Series(improvement, index=frame.index).groupby(
        frame["subject_id"].astype(str).to_numpy()
    ).mean()


def subject_bootstrap_interval(
    per_subject: pd.Series, n_boot: int = 2000, seed: int = SEED
) -> tuple[float, tuple[float, float], float]:
    values = per_subject.to_numpy(dtype=float)
    observed = float(np.mean(values))
    if n_boot <= 0:
        return observed, (float("nan"), float("nan")), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    return (
        observed,
        (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))),
        float(np.mean(draws)),
    )


def _endpoint_once(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    report_column: str,
    n_boot: int,
    seed: int,
) -> tuple[float, tuple[float, float], float, float, float, float]:
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    baseline_model = _fit_logistic(train, baseline_columns)
    augmented_model = _fit_logistic(train, baseline_columns + [report_column])
    baseline_scores = _scores(baseline_model, test, baseline_columns)
    augmented_scores = _scores(augmented_model, test, baseline_columns + [report_column])
    per_subject = paired_improvement(test, baseline_scores, augmented_scores)
    observed, interval, bootstrap_mean = subject_bootstrap_interval(per_subject, n_boot, seed)
    labels = test["y"].to_numpy(dtype=int)
    baseline_auroc = (
        float(roc_auc_score(labels, baseline_scores)) if len(np.unique(labels)) > 1 else float("nan")
    )
    augmented_auroc = (
        float(roc_auc_score(labels, augmented_scores)) if len(np.unique(labels)) > 1 else float("nan")
    )
    return (
        observed,
        interval,
        bootstrap_mean,
        float(np.mean((baseline_scores - labels) ** 2)),
        float(np.mean((augmented_scores - labels) ** 2)),
        baseline_auroc,
        augmented_auroc,
    )


def _shuffle_report_within_question(frame: pd.DataFrame, report_column: str, rng) -> pd.DataFrame:
    shuffled = frame.copy()
    for question, block in shuffled.groupby("question_id"):
        values = block[report_column].to_numpy(copy=True)
        shuffled.loc[block.index, report_column] = rng.permutation(values)
    return shuffled


def paired_brier_endpoint(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    report_column: str = "confidence",
    label_column: str = "y",
    n_boot: int = 2000,
    n_shuffle: int = 200,
    seed: int = SEED,
    provenance: dict | None = None,
) -> EndpointResult:
    """Run the endpoint and the within-question shuffle test."""

    required = {"subject_id", "question_id", "state", "split", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={label_column: "y"})
    observed, interval, bootstrap_mean, baseline_brier, augmented_brier, base_auc, aug_auc = (
        _endpoint_once(frame, baseline_columns, report_column, n_boot, seed)
    )
    rng = np.random.default_rng(seed + 1)
    null_values = []
    for _ in range(n_shuffle):
        shuffled = _shuffle_report_within_question(frame, report_column, rng)
        value, *_ = _endpoint_once(shuffled, baseline_columns, report_column, 0, seed)
        null_values.append(value)
    null = np.asarray(null_values, dtype=float)
    shuffle_p = float((1 + np.sum(null >= observed)) / (1 + len(null))) if len(null) else None
    return EndpointResult(
        n_subjects=int(frame[frame["split"] == "test"]["subject_id"].nunique()),
        n_observations=int((frame["split"] == "test").sum()),
        baseline_brier=baseline_brier,
        augmented_brier=augmented_brier,
        improvement=observed,
        improvement_ci=interval,
        bootstrap_mean=bootstrap_mean,
        baseline_auroc=base_auc,
        augmented_auroc=aug_auc,
        shuffle_mean=float(np.mean(null)) if len(null) else None,
        shuffle_p_value=shuffle_p,
        n_shuffle=n_shuffle,
        provenance=provenance or {},
    )


def hierarchical_success_estimates(
    frame: pd.DataFrame, draws_column: str = "n_draws", success_column: str = "n_success"
) -> pd.Series:
    """Beta-binomial posterior mean success per subject (plots only).

    The plan is explicit: shrunken estimates are useful for plots and must not
    be used as exact regressors.
    """

    grouped = frame.groupby("subject_id").agg(
        draws=(draws_column, "sum"), success=(success_column, "sum")
    )
    alpha, beta = 1.0, 1.0
    return (grouped["success"] + alpha) / (grouped["draws"] + alpha + beta)


def reliability_bins(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray | None = None) -> list[dict]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high)
        if high == edges[-1]:
            mask |= scores == high
        n = int(mask.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n": n,
                "mean_score": float(scores[mask].mean()) if n else None,
                "mean_label": float(labels[mask].mean()) if n else None,
            }
        )
    return rows
