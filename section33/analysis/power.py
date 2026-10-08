"""Power simulation for the frozen decision rules (plan section 10.2).

The estimator must work for the planned cohorts, where Track A has one answer
per subject.  A within-subject AUROC difference is undefined in that design, so
the variance of the pooled paired difference is estimated by resampling
subjects and recomputing the pooled metrics (``paired_development_distribution``).

The simulator then draws the *observed sample mean* of each difference from
``N(true effect, se^2)`` and builds the same interval the decision rule
receives.  Both the sampling variation and the interval width therefore scale
with the simulated number of subjects; drawing population-level effects with a
fixed interval (the earlier behaviour) does not.

The calibration guardrail is still represented by an explicit pass-probability
assumption rather than a full simulated calibration fit; that assumption is
recorded in every power table.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import PROTOCOL
from .difficulty import within_question_auroc
from .evaluation import decide_primary, decide_within_question


def paired_development_distribution(
    frame,
    score_a: str,
    score_b: str,
    label: str = "label",
    subject: str = "subject_id",
    question: str = "question_id",
    strata: tuple[str, ...] = ("relation", "stratum"),
    n_boot: int = PROTOCOL.bootstrap_samples,
    seed: int = PROTOCOL.bootstrap_seed,
) -> dict:
    """Subject bootstrap of the pooled paired AUROC and Brier differences.

    Subjects are resampled within the frozen relation/stratum cells and all of
    a subject's answers stay together.  Every draw recomputes the pooled
    metrics, so the returned arrays describe sampling at the development size
    even when each subject contributes one answer.  When ``question`` is
    present the per-draw within-question AUROC difference is returned as well.
    """

    subjects = frame[subject].astype(str).unique().tolist()
    empty = np.asarray([], dtype=float)
    if len(subjects) < 2:
        return {
            "delta_auc": empty,
            "delta_brier": empty,
            "delta_within": empty,
            "n_subjects": len(subjects),
            "n_boot": 0,
        }

    index_by_subject = {
        subject_id: frame.index[frame[subject].astype(str) == subject_id].to_numpy()
        for subject_id in subjects
    }
    strata_lookup: dict[str, tuple] = {}
    if strata and all(column in frame.columns for column in strata):
        first = frame.drop_duplicates(subject)
        for row in first.itertuples(index=False):
            strata_lookup[str(getattr(row, subject))] = tuple(
                getattr(row, column) for column in strata
            )
    groups: dict[tuple, list[str]] = {}
    for subject_id in subjects:
        groups.setdefault(strata_lookup.get(subject_id, ()), []).append(subject_id)

    rng = np.random.default_rng(seed)
    auc_deltas, brier_deltas, within_deltas = [], [], []
    for _ in range(n_boot):
        drawn: list[str] = []
        for key, subject_ids in groups.items():
            drawn.extend(rng.choice(subject_ids, size=len(subject_ids), replace=True))
        rows = np.concatenate([index_by_subject[subject_id] for subject_id in drawn])
        block = frame.loc[rows]
        y = block[label].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        a = np.clip(block[score_a].to_numpy(dtype=float), 0.0, 1.0)
        b = np.clip(block[score_b].to_numpy(dtype=float), 0.0, 1.0)
        auc_deltas.append(roc_auc_score(y, a) - roc_auc_score(y, b))
        brier_deltas.append(brier_score_loss(y, b) - brier_score_loss(y, a))
        if question in block.columns:
            within_a = within_question_auroc(a, y, block[question].to_numpy())
            within_b = within_question_auroc(b, y, block[question].to_numpy())
            if (
                within_a["within_question_auroc"] is not None
                and within_b["within_question_auroc"] is not None
            ):
                within_deltas.append(
                    within_a["within_question_auroc"] - within_b["within_question_auroc"]
                )
    return {
        "delta_auc": np.asarray(auc_deltas, dtype=float),
        "delta_brier": np.asarray(brier_deltas, dtype=float),
        "delta_within": np.asarray(within_deltas, dtype=float),
        "n_subjects": len(subjects),
        "n_boot": len(auc_deltas),
    }


def development_variance(distribution: dict) -> dict:
    """Standard errors, means, and correlations at the development size."""

    auc = np.asarray(distribution.get("delta_auc", []), dtype=float)
    brier = np.asarray(distribution.get("delta_brier", []), dtype=float)
    within = np.asarray(distribution.get("delta_within", []), dtype=float)
    result = {
        "n_subjects": int(distribution.get("n_subjects", 0)),
        "n_boot": int(len(auc)),
        "se_auc": float("nan"),
        "se_brier": float("nan"),
        "delta_auc_mean": float("nan"),
        "delta_brier_mean": float("nan"),
        "corr": float("nan"),
        "se_within": float("nan"),
        "delta_within_mean": float("nan"),
    }
    if len(auc) < 2 or len(brier) < 2:
        return result
    result.update(
        se_auc=float(np.std(auc, ddof=1)),
        se_brier=float(np.std(brier, ddof=1)),
        delta_auc_mean=float(np.mean(auc)),
        delta_brier_mean=float(np.mean(brier)),
        corr=float(np.corrcoef(auc, brier)[0, 1]) if len(auc) > 2 else 0.0,
    )
    if len(within) >= 2:
        result["se_within"] = float(np.std(within, ddof=1))
        result["delta_within_mean"] = float(np.mean(within))
    return result


def simulate_paired_decision(
    n_subjects: int,
    delta_auc_mean: float,
    delta_brier_mean: float,
    se_auc: float,
    se_brier: float,
    correlation: float = 0.8,
    n_sim: int = 2000,
    seed: int = 0,
    calibration_ok_probability: float = 0.9,
) -> dict:
    """Probability of each primary decision at ``n_subjects``.

    ``se_auc`` and ``se_brier`` are the standard errors **at ``n_subjects``**.
    Each simulated experiment draws the observed sample mean from
    ``N(true effect, se^2)`` and builds the same 95% interval the decision rule
    receives.  The calibration guardrail is an explicit pass-probability
    assumption, recorded in the result.
    """

    rng = np.random.default_rng(seed)
    covariance = [
        [se_auc**2, correlation * se_auc * se_brier],
        [correlation * se_auc * se_brier, se_brier**2],
    ]
    draws = rng.multivariate_normal([delta_auc_mean, delta_brier_mean], covariance, size=n_sim)
    verdicts = {
        "useful_improvement_supported": 0,
        "useful_improvement_ruled_out": 0,
        "harm_supported": 0,
        "inconclusive": 0,
    }
    half_width_auc = 1.96 * se_auc
    half_width_brier = 1.96 * se_brier
    for index in range(n_sim):
        auc = float(draws[index, 0])
        brier = float(draws[index, 1])
        interval = (auc - half_width_auc, auc + half_width_auc)
        brier_interval = (brier - half_width_brier, brier + half_width_brier)
        calibration_ok = rng.random() < calibration_ok_probability
        decision = decide_primary(
            {
                "delta_auc": auc,
                "delta_auc_interval": list(interval),
                "delta_brier": brier,
                "delta_brier_interval": list(brier_interval),
            },
            calibration_error_increase=(0.0, 0.004 if calibration_ok else 0.02),
        )
        verdicts[decision.verdict] = verdicts.get(decision.verdict, 0) + 1
    return {
        "n_subjects": n_subjects,
        "delta_auc_mean": delta_auc_mean,
        "delta_brier_mean": delta_brier_mean,
        "se_auc": se_auc,
        "se_brier": se_brier,
        "calibration_ok_probability": calibration_ok_probability,
        **{f"p_{key}": value / n_sim for key, value in verdicts.items()},
    }


def simulate_within_question_decision(
    n_subjects: int,
    delta_mean: float,
    se: float,
    margin: float | None = None,
    n_sim: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Probability of each within-question verdict at ``n_subjects``."""

    margin = margin or PROTOCOL.margins.within_question_auroc
    rng = np.random.default_rng(seed)
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    verdicts = {
        "within_question_gain_supported": 0,
        "within_question_gain_ruled_out": 0,
        "within_question_inconclusive": 0,
    }
    for _ in range(n_sim):
        delta = float(rng.normal(delta_mean, se))
        interval = (delta - z * se, delta + z * se)
        decision = decide_within_question(interval, (-1.0, 0.0), margin=margin)
        verdicts[decision.verdict] = verdicts.get(decision.verdict, 0) + 1
    return {
        "n_subjects": n_subjects,
        "delta_mean": delta_mean,
        "se": se,
        "margin": margin,
        **{f"p_{key}": value / n_sim for key, value in verdicts.items()},
    }


def default_test_grid() -> list[int]:
    return [800, 1600, 2400, 4000, 6000]


def power_table(
    se_auc: float,
    se_brier: float,
    n_development: int,
    correlation: float = 0.8,
    useful_effect: tuple[float, float] = (0.040, 0.010),
    negligible_effect: tuple[float, float] = (0.0, 0.0),
    n_sim: int = 2000,
    seed: int = 0,
) -> dict:
    """Power table over the frozen test-size grid for both alternatives.

    ``se_auc`` and ``se_brier`` are the development-size standard errors; each
    grid point rescales them by ``sqrt(n_development / n_subjects)``.
    """

    rows = []
    for n_subjects in default_test_grid():
        scale = float(np.sqrt(n_development / n_subjects))
        se_a = se_auc * scale
        se_b = se_brier * scale
        useful = simulate_paired_decision(
            n_subjects, useful_effect[0], useful_effect[1], se_a, se_b,
            correlation=correlation, n_sim=n_sim, seed=seed,
        )
        negligible = simulate_paired_decision(
            n_subjects, negligible_effect[0], negligible_effect[1], se_a, se_b,
            correlation=correlation, n_sim=n_sim, seed=seed + 1,
        )
        rows.append({"n_subjects": n_subjects, "useful": useful, "negligible": negligible})
    return {
        "assumptions": {
            "se_auc_at_development": se_auc,
            "se_brier_at_development": se_brier,
            "n_development": n_development,
            "correlation": correlation,
            "useful_effect": list(useful_effect),
            "negligible_effect": list(negligible_effect),
            "n_sim": n_sim,
            "bootstrap_samples": PROTOCOL.bootstrap_samples,
        },
        "rows": rows,
    }


def within_question_power_table(
    se_within: float,
    n_development: int,
    useful_effect: float = 0.060,
    negligible_effect: float = 0.0,
    n_sim: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Power table for the within-question claim over the frozen test grid."""

    rows = []
    for n_subjects in default_test_grid():
        scale = float(np.sqrt(n_development / n_subjects))
        useful = simulate_within_question_decision(
            n_subjects, useful_effect, se_within * scale, n_sim=n_sim, seed=seed,
            confidence=confidence,
        )
        negligible = simulate_within_question_decision(
            n_subjects, negligible_effect, se_within * scale, n_sim=n_sim, seed=seed + 1,
            confidence=confidence,
        )
        rows.append({"n_subjects": n_subjects, "useful": useful, "negligible": negligible})
    return {
        "assumptions": {
            "se_within_at_development": se_within,
            "n_development": n_development,
            "useful_effect": useful_effect,
            "negligible_effect": negligible_effect,
            "margin": PROTOCOL.margins.within_question_auroc,
            "confidence": confidence,
            "n_sim": n_sim,
        },
        "rows": rows,
    }
