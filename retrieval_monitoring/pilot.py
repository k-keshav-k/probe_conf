"""S3: the 40-subject selectivity pilot.

Question: does the relation-map intervention damage the queried relation more
than the controls?  The pilot uses 40 development subjects, two relations, four
states, and eight answer draws per cell.  Damage is measured from the clean
state, paired within subject, with subject-level bootstrap intervals.

The confidence-change contrasts are specificity diagnostics.  They are
interpreted alongside the actual success changes, never instead of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("clean", "relevant", "other", "random")


@dataclass
class PilotThresholds:
    min_relevant_damage: float = 0.05
    min_selectivity: float = 0.03


def load_rows(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no rows in {path}")
    return frame


def select_subject_pairs(
    cohort: pd.DataFrame,
    n_subjects: int,
    relations: tuple[str, str],
    split: str = "dev",
) -> list[dict]:
    """Subjects with one question in each of two relations.

    The cohort used by the Section 3.3 experiment has one question per subject,
    so it cannot supply pairs.  A pilot cohort must have two questions for the
    same subject.  The error message says so explicitly.
    """

    block = cohort[cohort["split"] == split]
    pairs: list[dict] = []
    for subject_id, subject_block in block.groupby("subject_id"):
        entries = []
        for relation in relations:
            match = subject_block[subject_block["relation"] == relation]
            if match.empty:
                continue
            row = match.iloc[0]
            entries.append(
                {
                    "relation": relation,
                    "question_id": str(row["question_id"]),
                    "question": str(row["question"]),
                    "reference_answers": list(row["reference_answers"])
                    if isinstance(row["reference_answers"], (list, tuple, np.ndarray))
                    else [str(row["reference_answers"])],
                }
            )
        if len(entries) == 2:
            pairs.append(
                {
                    "subject_id": str(subject_id),
                    "name": str(subject_block.iloc[0]["subject_name"]),
                    "relations": entries,
                }
            )
        if len(pairs) >= n_subjects:
            break
    if len(pairs) < n_subjects:
        raise ValueError(
            f"found {len(pairs)} subjects with both relations in split {split}; "
            f"need {n_subjects}. Build a pilot cohort with two questions per subject."
        )
    return pairs


def success_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean success per (subject, queried relation, state)."""

    answers = rows[rows["kind"] == "answer"].copy()
    if answers.empty:
        raise ValueError("no answer rows")
    table = (
        answers.groupby(["subject_id", "queried_relation", "state"])["label"]
        .mean()
        .rename("success")
        .reset_index()
    )
    return table


def confidence_table(rows: pd.DataFrame) -> pd.DataFrame:
    confidence = rows[rows["kind"] == "confidence"].copy()
    if confidence.empty:
        return pd.DataFrame(columns=["subject_id", "queried_relation", "state", "confidence"])
    return (
        confidence.groupby(["subject_id", "queried_relation", "state"])["confidence"]
        .mean()
        .reset_index()
    )


def _paired_changes(table: pd.DataFrame, value: str, baseline: str = "clean") -> pd.DataFrame:
    wide = table.pivot_table(
        index=["subject_id", "queried_relation"], columns="state", values=value
    ).reset_index()
    if baseline not in wide.columns:
        raise ValueError(f"missing baseline state {baseline}")
    changes = {"subject_id": wide["subject_id"], "queried_relation": wide["queried_relation"]}
    for state in STATES:
        if state == baseline:
            continue
        changes[f"change_{state}"] = wide[baseline] - wide[state]
    return pd.DataFrame(changes)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return float(np.mean(values)), (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def selectivity_report(
    rows: pd.DataFrame,
    thresholds: PilotThresholds | None = None,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    thresholds = thresholds or PilotThresholds()
    table = success_table(rows)
    changes = _paired_changes(table, "success")
    # Average the two queried relations within each subject.
    per_subject = changes.groupby("subject_id").mean(numeric_only=True)
    report: dict = {"thresholds": thresholds.__dict__, "n_subjects": int(len(per_subject))}
    for state in ("relevant", "other", "random"):
        column = f"change_{state}"
        values = per_subject[column].to_numpy(dtype=float)
        mean, interval = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
        report[f"damage_{state}"] = {"mean": mean, "ci": list(interval)}
    for control in ("other", "random"):
        values = (
            per_subject["change_relevant"] - per_subject[f"change_{control}"]
        ).to_numpy(dtype=float)
        mean, interval = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + 1)
        report[f"selectivity_vs_{control}"] = {"mean": mean, "ci": list(interval)}
    clean_success = table[table["state"] == "clean"]["success"]
    report["clean_success"] = {
        "mean": float(clean_success.mean()),
        "ceiling_share": float((clean_success >= 0.99).mean()),
        "floor_share": float((clean_success <= 0.01).mean()),
    }
    confidence = confidence_table(rows)
    if not confidence.empty:
        confidence_changes = _paired_changes(confidence, "confidence")
        confidence_subject = confidence_changes.groupby("subject_id").mean(numeric_only=True)
        report["confidence_diagnostic"] = {}
        for state in ("relevant", "other", "random"):
            column = f"change_{state}"
            if column in confidence_subject.columns:
                values = confidence_subject[column].to_numpy(dtype=float)
                mean, interval = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + 2)
                report["confidence_diagnostic"][state] = {"mean": mean, "ci": list(interval)}
    relevant = report["damage_relevant"]["mean"]
    selectivity = report["selectivity_vs_random"]["mean"]
    report["passed"] = bool(
        relevant >= thresholds.min_relevant_damage
        and selectivity >= thresholds.min_selectivity
    )
    report["reason"] = (
        "selective damage detected"
        if report["passed"]
        else "selectivity gate not met; revise the intervention and repeat development"
    )
    return report


def save_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
