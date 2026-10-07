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


def load_split(
    paths: dict[str, Path],
    model_slug: str,
    track: str,
    splits: tuple[str, ...] = ("train", "dev", "calibration", "test"),
) -> pd.DataFrame:
    """Join labels, features, and confidence for one track and model."""

    suffix = f"__{model_slug}.jsonl"
    labels = _read(paths["labels"] / f"track{track}_answers{suffix}")
    features = _read(paths["features"] / f"track{track}_features{suffix}")
    confidence = _read(paths["labels"] / f"track{track}_confidence{suffix}")
    if labels.empty or features.empty:
        return pd.DataFrame()
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


def metadata_features(frame: pd.DataFrame, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Difficulty block D: relation one-hot, popularity, lengths, missingness.

    Only columns present in the frame are used.  The plan requires that any
    proposed metadata not yet collected is added before use.
    """

    import numpy as np

    available = columns or ("relation", "stratum", "content_length", "conf_missing")
    parts = []
    if "relation" in frame.columns:
        parts.append(pd.get_dummies(frame["relation"], prefix="rel").astype(float))
    if "stratum" in frame.columns:
        parts.append(pd.get_dummies(frame["stratum"], prefix="stratum").astype(float))
    for column in ("popularity", "content_length", "prompt_tokens", "conf"):
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
