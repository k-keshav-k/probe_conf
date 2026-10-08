"""Power simulation for the frozen decision rules (plan section 10.2).

Two simulators are provided:

- ``simulate_paired_decision``: subject-level paired normal deltas with a
  stated mean effect and standard deviation.  Use it to map effect sizes and
  sample sizes to the probability of each verdict.
- ``development_variance``: estimate the per-subject standard deviations of the
  metric differences from development predictions.  Never use the
  historical gold-dependent gain as the assumed effect.
"""

from __future__ import annotations

import numpy as np

from ..config import PROTOCOL
from .evaluation import decide_primary


def simulate_paired_decision(
    n_subjects: int,
    delta_auc_mean: float,
    delta_brier_mean: float,
    sd_auc: float,
    sd_brier: float,
    correlation: float = 0.8,
    n_sim: int = 2000,
    seed: int = 0,
    calibration_ok_probability: float = 0.9,
) -> dict:
    """Probability of each primary decision for a given subject-level effect."""

    rng = np.random.default_rng(seed)
    covariance = [
        [sd_auc**2, correlation * sd_auc * sd_brier],
        [correlation * sd_auc * sd_brier, sd_brier**2],
    ]
    draws = rng.multivariate_normal([delta_auc_mean, delta_brier_mean], covariance, size=n_sim)
    verdicts = {"useful_improvement_supported": 0, "useful_improvement_ruled_out": 0,
                "harm_supported": 0, "inconclusive": 0}
    half_width_auc = 1.96 * sd_auc / np.sqrt(max(1, n_subjects))
    half_width_brier = 1.96 * sd_brier / np.sqrt(max(1, n_subjects))
    for index in range(n_sim):
        auc = draws[index, 0]
        brier = draws[index, 1]
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
        **{f"p_{key}": value / n_sim for key, value in verdicts.items()},
    }


def development_variance(
    deltas_auc: np.ndarray,
    deltas_brier: np.ndarray,
) -> dict:
    """Per-subject standard deviations of the paired metric differences.

    ``simulate_paired_decision`` expects the SD of the per-subject deltas and
    turns it into a standard error with ``sd / sqrt(n_subjects)``.  Bootstrapping
    the means here would return a standard error instead and shrink every
    simulated interval by another factor of ``sqrt(n)``.
    """

    deltas_auc = np.asarray(deltas_auc, dtype=float)
    deltas_brier = np.asarray(deltas_brier, dtype=float)
    n = int(min(len(deltas_auc), len(deltas_brier)))
    if n < 2 or len(deltas_auc) != len(deltas_brier):
        # Not estimable: no mixed-label subjects, or the two arrays are not
        # paired (for example a cohort with one question per subject).
        return {
            "sd_auc": float("nan"),
            "sd_brier": float("nan"),
            "corr": 0.0,
            "n_subjects": n,
        }
    return {
        "sd_auc": float(np.std(deltas_auc, ddof=1)),
        "sd_brier": float(np.std(deltas_brier, ddof=1)),
        "corr": float(np.corrcoef(deltas_auc, deltas_brier)[0, 1]),
        "n_subjects": n,
    }


def default_test_grid() -> list[int]:
    return [800, 1600, 2400, 4000, 6000]


def power_table(
    sd_auc: float,
    sd_brier: float,
    correlation: float = 0.8,
    useful_effect: tuple[float, float] = (0.040, 0.010),
    negligible_effect: tuple[float, float] = (0.0, 0.0),
    n_sim: int = 2000,
    seed: int = 0,
) -> dict:
    """Power table over the frozen test-size grid for both alternatives."""

    rows = []
    for n_subjects in default_test_grid():
        useful = simulate_paired_decision(
            n_subjects, useful_effect[0], useful_effect[1], sd_auc, sd_brier,
            correlation=correlation, n_sim=n_sim, seed=seed,
        )
        negligible = simulate_paired_decision(
            n_subjects, negligible_effect[0], negligible_effect[1], sd_auc, sd_brier,
            correlation=correlation, n_sim=n_sim, seed=seed + 1,
        )
        rows.append({"n_subjects": n_subjects, "useful": useful, "negligible": negligible})
    return {
        "assumptions": {
            "sd_auc": sd_auc,
            "sd_brier": sd_brier,
            "correlation": correlation,
            "useful_effect": list(useful_effect),
            "negligible_effect": list(negligible_effect),
            "n_sim": n_sim,
            "bootstrap_samples": PROTOCOL.bootstrap_samples,
        },
        "rows": rows,
    }
