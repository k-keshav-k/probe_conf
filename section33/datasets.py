"""Loading and joining run artefacts for development and evaluation.

The labels and features interfaces are separate files.  This module joins them
on ``(question_id, track, answer_index)`` for analysis and drops reference
columns from the analysis frame, because analysis code must not consume the
reference answer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import read_jsonl

REFERENCE_COLUMNS = ("reference_answers",)


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.DataFrame(read_jsonl(Path(path)))


def _dedupe(frame: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Drop duplicate rows, tolerating list-valued key columns."""

    if frame.empty:
        return frame
    work = frame.copy()
    for column in subset:
        if column in work.columns:
            work[column] = work[column].map(
                lambda value: tuple(value) if isinstance(value, list) else value
            )
    return work.drop_duplicates(subset=[c for c in subset if c in work.columns], keep="first")


def load_split(
    paths: dict[str, Path],
    model_slug: str,
    track: str,
    splits: tuple[str, ...] = ("train", "dev", "calibration", "test"),
) -> pd.DataFrame:
    """Join labels, features, and confidence for one track and model.

    Sharded collections write one file group per shard.  All groups that match
    ``track{track}_*__{model_slug}*`` are merged.  Track B keys on the candidate
    itself, because ``answer_index`` is batch-local and can differ when the same
    question was collected under a different shard tag.
    """

    def _glob_read(folder: str, pattern: str) -> pd.DataFrame:
        frames = [_read(path) for path in sorted(Path(folder).glob(pattern))]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    labels = _glob_read(paths["labels"], f"track{track}_answers__{model_slug}*.jsonl")
    features = _glob_read(paths["features"], f"track{track}_features__{model_slug}*.jsonl")
    confidence = _glob_read(paths["labels"], f"track{track}_confidence__{model_slug}*.jsonl")
    if labels.empty or features.empty:
        return pd.DataFrame()
    if track.upper() == "B":
        labels = _dedupe(labels, ["question_id", "track", "candidate_text"])
        features = _dedupe(features, ["question_id", "track", "candidate_token_ids", "position"])
    else:
        labels = _dedupe(labels, ["question_id", "track", "answer_index"])
        features = _dedupe(features, ["question_id", "track", "answer_index", "position"])
    confidence = _dedupe(confidence, ["question_id", "track", "answer_index"])
    if track.upper() == "A":
        feature_primary = features[features["position"] == "last_prompt"]
    else:
        feature_primary = features[features["position"] == "answer_last_content"]
    feature_columns = [c for c in feature_primary.columns if c.startswith("f__") or c in
                       ("subject_id", "question_id", "split", "track", "answer_index", "generation_seed",
                        "position", "candidate_token_ids", "content_length", "multiplicity")]
    merged = labels.merge(
        feature_primary[feature_columns],
        on=["subject_id", "question_id", "split", "track", "answer_index"],
        how="inner",
        suffixes=("", "_feat"),
    )
    if not confidence.empty:
        merged = merged.merge(
            confidence[["question_id", "track", "answer_index", "value", "parse_ok", "raw_text"]].rename(
                columns={"value": "conf", "parse_ok": "conf_parse_ok", "raw_text": "conf_raw"}
            ),
            on=["question_id", "track", "answer_index"],
            how="left",
        )
    else:
        merged["conf"] = pd.NA
        merged["conf_parse_ok"] = False
        merged["conf_raw"] = ""
    merged = merged[merged["split"].isin(splits)].reset_index(drop=True)
    for column in REFERENCE_COLUMNS:
        if column in merged.columns:
            merged = merged.drop(columns=[column])
    return merged


def metadata_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Difficulty block D: relation one-hot, popularity, lengths, missingness.

    The stated confidence score is deliberately excluded.  It belongs to the
    separate V comparator (plan section 6.1: B = D + P + H).  Only the
    missingness indicators for confidence are kept here.
    """

    parts = []
    if "relation" in frame.columns:
        parts.append(pd.get_dummies(frame["relation"], prefix="rel").astype(float))
    if "stratum" in frame.columns:
        parts.append(pd.get_dummies(frame["stratum"], prefix="stratum").astype(float))
    for column in ("popularity", "content_length", "prompt_tokens"):
        if column in frame.columns:
            parts.append(frame[[column]].astype(float).rename(columns={column: f"meta_{column}"}))
    if "conf" in frame.columns:
        parts.append(
            pd.DataFrame(
                {
                    "conf_missing": frame["conf"].isna().astype(float),
                    "conf_parse_failed": (~frame.get("conf_parse_ok", pd.Series(True, index=frame.index)).astype(bool)).astype(float),
                }
            )
        )
    if not parts:
        return pd.DataFrame(index=frame.index)
    out = pd.concat(parts, axis=1)
    return out.fillna(0.0)


def expand_by_multiplicity(frame: pd.DataFrame, column: str = "multiplicity") -> pd.DataFrame:
    """Repeat each row by its sampling multiplicity.

    Track B caches duplicate candidate strings but keeps their sampling
    multiplicity.  Expanding before a metric makes every sampled answer count
    once, as the plan requires.
    """

    if frame.empty or column not in frame.columns:
        return frame.reset_index(drop=True)
    weights = pd.to_numeric(frame[column], errors="coerce").fillna(1).astype(int).clip(lower=1)
    return frame.loc[frame.index.repeat(weights)].reset_index(drop=True)
