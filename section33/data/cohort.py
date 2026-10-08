"""Cohort construction, exclusion, quotas, and split manifests.

The plan (section 4) requires:

- one primary question per subject,
- exclusion of every previously used subject,
- equal relation quotas and equal popularity-quintile quotas within relation,
- a dated source with revisions recorded,
- subject-grouped ranges that stay in one split,
- a hashed manifest.

This module works on a table with one row per candidate question.  Accepted
formats: Parquet, CSV, JSONL, or an Arrow file.  Required columns (aliases in
brackets are accepted):

- ``question_id``  [id]
- ``subject_id``   [subj_id, qid]
- ``subject_name`` [subj]
- ``relation``     [prop]
- ``question``
- ``reference_answers`` [possible_answers, o_aliases, obj]  (list or JSON string)
- ``popularity``   [s_pop]  (may be missing; an explicit stratum is used)
- ``source_revision`` (optional; filled with the snapshot date when absent)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MISSING_POPULARITY_STRATUM, POPULARITY_BINS, RELATIONS
from ..utils import log, manifest_hash, sha256_file, write_json

_COLUMN_ALIASES = {
    "question_id": ["question_id", "id"],
    "subject_id": ["subject_id", "subj_id", "qid"],
    "subject_name": ["subject_name", "subj"],
    "relation": ["relation", "prop"],
    "question": ["question"],
    "reference_answers": ["reference_answers", "possible_answers", "o_aliases", "obj"],
    "popularity": ["popularity", "s_pop"],
    "source_revision": ["source_revision", "revision"],
}


@dataclass
class CohortQuestion:
    question_id: str
    subject_id: str
    subject_name: str
    relation: str
    question: str
    reference_answers: list[str]
    split: str
    popularity: float | None = None
    stratum: str = MISSING_POPULARITY_STRATUM
    source_revision: str = ""
    family_id: str = ""

    def to_row(self) -> dict:
        row = self.__dict__.copy()
        row["reference_answers"] = json.dumps(self.reference_answers)
        return row


# ---------------------------------------------------------------------------
# Source loading and normalisation
# ---------------------------------------------------------------------------
def load_source_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".arrow":
        try:
            from datasets import Dataset

            return Dataset.from_file(str(path)).to_pandas()
        except Exception:
            import pyarrow as pa

            try:
                return pa.ipc.open_file(path).read_pandas().to_pandas()
            except pa.lib.ArrowInvalid:
                return pa.ipc.open_stream(path).read_pandas().to_pandas()
    raise ValueError(f"unsupported source format: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                out[canonical] = df[alias]
                break
        else:
            if canonical in ("source_revision",):
                out[canonical] = ""
            elif canonical == "popularity":
                out[canonical] = np.nan
            else:
                raise KeyError(f"missing source column for {canonical}: {aliases}")
    out["question_id"] = out["question_id"].astype(str)
    out["subject_id"] = out["subject_id"].astype(str)
    out["relation"] = (
        out["relation"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    out["source_revision"] = out["source_revision"].fillna("").astype(str)
    return out


def parse_reference_answers(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        pass
    return [text] if text.strip() else []


# ---------------------------------------------------------------------------
# Exclusion and selection
# ---------------------------------------------------------------------------
def load_seen_subjects(label_paths: list[Path]) -> set[str]:
    """Union of subject IDs from every historical label file."""

    seen: set[str] = set()
    for path in label_paths:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ("subj_id", "subject_id"))
        column = "subj_id" if "subj_id" in df.columns else "subject_id"
        seen.update(df[column].astype(str))
    return seen


def exclude_seen(df: pd.DataFrame, seen_subjects: set[str]) -> pd.DataFrame:
    return df[~df["subject_id"].astype(str).isin(seen_subjects)].reset_index(drop=True)


def one_question_per_subject(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    ranked = df.copy()
    ranked["_rank"] = ranked.apply(
        lambda row: int(manifest_hash([str(seed), row["subject_id"], row["question_id"]])[:12], 16),
        axis=1,
    )
    ranked = ranked.sort_values(["subject_id", "_rank"])
    picked = ranked.groupby("subject_id", as_index=False).first()
    return picked.drop(columns=["_rank"]).reset_index(drop=True)


def assign_popularity_strata(df: pd.DataFrame, bins: int = POPULARITY_BINS) -> pd.DataFrame:
    """Quintile strata inside each relation; explicit missing stratum."""

    out = df.copy()
    out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce")
    out["stratum"] = MISSING_POPULARITY_STRATUM
    for relation, index in out.groupby("relation").groups.items():
        values = out.loc[index, "popularity"]
        valid = values.dropna()
        if len(valid) < bins:
            out.loc[index, "stratum"] = MISSING_POPULARITY_STRATUM
            continue
        edges = np.quantile(np.log10(valid + 1.0), np.linspace(0, 1, bins + 1))
        edges[-1] += 1e-9
        bucketed = np.searchsorted(edges, np.log10(values.fillna(0) + 1.0), side="right") - 1
        bucketed = np.clip(bucketed, 0, bins - 1).astype(int).astype(str)
        bucketed[values.isna()] = MISSING_POPULARITY_STRATUM
        out.loc[index, "stratum"] = bucketed
    return out


def balanced_sample(
    df: pd.DataFrame,
    per_relation: int,
    seed: int = 0,
    strata_column: str = "stratum",
) -> pd.DataFrame:
    """Equal relation quotas, then equal stratum quotas inside each relation."""

    rng = np.random.default_rng(seed)
    picked: list[pd.DataFrame] = []
    for relation in RELATIONS:
        block = df[df["relation"] == relation]
        if block.empty:
            continue
        strata = sorted(block[strata_column].unique())
        per_stratum = max(1, per_relation // max(1, len(strata)))
        for stratum in strata:
            cell = block[block[strata_column] == stratum]
            take = min(len(cell), per_stratum)
            if take:
                indices = rng.choice(len(cell), size=take, replace=False)
                picked.append(cell.iloc[indices])
    if not picked:
        return df.head(0).copy()
    out = pd.concat(picked).reset_index(drop=True)
    return out


def assign_splits(
    df: pd.DataFrame,
    sizes: dict,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic subject-level split assignment.

    Order subjects by a hash of ``seed`` and the subject ID, then fill
    train -> development -> calibration -> test.  All rows of a subject stay
    together.
    """

    subjects = sorted(df["subject_id"].unique())
    order = sorted(subjects, key=lambda s: manifest_hash([str(seed), s]))
    assignment: dict[str, str] = {}
    plan = [
        ("train", sizes["train"]),
        ("dev", sizes["dev"]),
        ("calibration", sizes["calibration"]),
        ("test", sizes["test"]),
    ]
    cursor = 0
    for name, count in plan:
        for subject in order[cursor : cursor + count]:
            assignment[subject] = name
        cursor += count
    # Any subject past the planned sizes goes to test (the cap stage).
    for subject in order[cursor:]:
        assignment[subject] = "test"
    out = df.copy()
    out["split"] = out["subject_id"].map(assignment)
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_manifest(df: pd.DataFrame, source: Path, seed: int, extra: dict | None = None) -> dict:
    counts = (
        df.groupby(["split", "relation", "stratum"]).size().rename("count").reset_index()
    )
    manifest = {
        "source": str(source),
        "source_hash": sha256_file(source) if Path(source).exists() else None,
        "seed": seed,
        "n_questions": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "subject_hash": manifest_hash(df["subject_id"].unique()),
        "counts": counts.to_dict(orient="records"),
        "split_subject_hashes": {
            split: manifest_hash(block["subject_id"].unique())
            for split, block in df.groupby("split")
        },
        "split_hashes": {
            split: manifest_hash(block["question_id"])
            for split, block in df.groupby("split")
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def verify_disjoint(df: pd.DataFrame) -> dict:
    """Raise if any subject appears in more than one split."""

    by_subject = df.groupby("subject_id")["split"].nunique()
    bad = by_subject[by_subject > 1]
    if len(bad):
        raise ValueError(f"subjects in multiple splits: {bad.index.tolist()[:5]}")
    overlaps = {}
    splits = sorted(df["split"].unique())
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            left_ids = set(df.loc[df["split"] == left, "question_id"])
            right_ids = set(df.loc[df["split"] == right, "question_id"])
            overlaps[f"{left}__{right}"] = len(left_ids & right_ids)
    return overlaps


def validate_cohort(
    df: pd.DataFrame,
    sizes: dict,
    *,
    mode: str = "pilot",
    relations: tuple[str, ...] = RELATIONS,
    excluded_subjects: int = 0,
) -> dict:
    """Check the cohort against the declared design.

    ``pilot`` records every problem as a warning and lets the run continue;
    ``confirmatory`` raises, so an undersized, unbalanced, or incompletely
    excluded cohort can never be used for a final claim (review F7).
    """

    problems: list[str] = []
    actual = df["split"].value_counts().to_dict()
    for split, requested in sizes.items():
        if actual.get(split, 0) != requested:
            problems.append(f"split {split}: requested {requested}, actual {actual.get(split, 0)}")
    present = set(df["relation"].unique())
    for relation in relations:
        if relation not in present:
            problems.append(f"relation {relation}: absent")
    relation_counts = df["relation"].value_counts().to_dict()
    if len(present) == len(relations):
        values = [relation_counts.get(relation, 0) for relation in relations]
        if min(values) != max(values):
            problems.append(f"relation counts unbalanced: {relation_counts}")
    if excluded_subjects == 0:
        problems.append("no excluded subjects: the historical exclusion list is empty")
    if mode == "confirmatory":
        missing_share = (
            float((df["stratum"] == MISSING_POPULARITY_STRATUM).mean()) if len(df) else 1.0
        )
        if missing_share > 0.10:
            problems.append(
                f"{missing_share:.0%} of subjects have no popularity stratum; "
                "join the frozen pageview dump before building the cohort"
            )
    report = {"mode": mode, "problems": problems, "passed": not problems}
    if problems and mode == "confirmatory":
        raise ValueError("cohort does not meet the confirmatory design: " + "; ".join(problems))
    return report


def build_cohort(
    source: Path,
    seen_subjects: set[str],
    *,
    per_relation: int,
    seed: int = 0,
    source_revision: str = "",
    allow_seen: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Full cohort pipeline on an existing source table."""

    raw = normalize_columns(load_source_table(source))
    raw = raw[raw["relation"].isin(RELATIONS)].reset_index(drop=True)
    if not allow_seen:
        raw = exclude_seen(raw, seen_subjects)
    raw["reference_answers"] = raw["reference_answers"].map(parse_reference_answers)
    raw = raw[raw["reference_answers"].map(len) > 0].reset_index(drop=True)
    raw = one_question_per_subject(raw, seed=seed)
    raw = assign_popularity_strata(raw)
    log(f"fresh pool: {len(raw)} questions, {raw['subject_id'].nunique()} subjects")
    sampled = balanced_sample(raw, per_relation=per_relation, seed=seed)
    if source_revision:
        sampled["source_revision"] = source_revision
    sampled = assign_splits(
        sampled,
        sizes={"train": 0, "dev": 0, "calibration": 0, "test": len(sampled)},
        seed=seed,
    )  # placeholder; real sizes are provided by the caller in freeze_splits
    return sampled, build_manifest(sampled, source, seed)


def freeze_splits(df: pd.DataFrame, sizes: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Assign the frozen split sizes and return the split manifest."""

    out = assign_splits(df, sizes, seed)
    manifest = {
        "seed": seed,
        "sizes_requested": sizes,
        "sizes_actual": out["split"].value_counts().to_dict(),
        "subject_hash": manifest_hash(out["subject_id"].unique()),
        "split_subject_hashes": {
            split: manifest_hash(block["subject_id"].unique())
            for split, block in out.groupby("split")
        },
    }
    verify_disjoint(out)
    return out, manifest
