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


def _has_stable_keys(frame: pd.DataFrame) -> bool:
    return (
        "candidate_key" in frame.columns
        and len(frame) > 0
        and frame["candidate_key"].notna().all()
        and (frame["candidate_key"].astype(str) != "").all()
    )


def load_split(
    paths: dict[str, Path],
    model_slug: str,
    track: str,
    splits: tuple[str, ...] = ("train", "dev", "calibration", "test"),
    expected_samples: int | None = None,
) -> pd.DataFrame:
    """Join labels, features, and confidence for one track and model.

    Sharded collections write one file group per shard.  All groups that match
    ``track{track}_*__{model_slug}*`` are merged.  New collections carry a
    stable ``candidate_key`` (model, revision, and exact token IDs) in every
    interface; legacy rows without keys fall back to the run-local
    ``answer_index``.  The join is validated: unmatched rows and duplicate keys
    raise instead of silently dropping samples.

    ``expected_samples`` (Track B) rejects any completed question that does not
    retain exactly that many draws after the join.
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

    stable = _has_stable_keys(labels) and _has_stable_keys(features)
    key_columns = ["question_id", "track", "candidate_key"] if stable else ["question_id", "track", "answer_index"]
    labels = _dedupe(labels, key_columns)
    features = _dedupe(features, key_columns + ["position"])

    if track.upper() == "A":
        feature_primary = features[features["position"] == "last_prompt"]
    else:
        feature_primary = features[features["position"] == "answer_last_content"]
    feature_columns = [c for c in feature_primary.columns if c.startswith("f__") or c in
                       ("subject_id", "question_id", "split", "track", "answer_index", "generation_seed",
                        "position", "candidate_token_ids", "candidate_key", "content_length", "multiplicity")]
    merged = labels.merge(
        feature_primary[feature_columns],
        on=key_columns,
        how="left",
        suffixes=("", "_feat"),
        indicator=True,
    )
    unmatched = int((merged["_merge"] == "left_only").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} {track} label rows have no matching feature row; "
            "check the candidate keys and collection completeness"
        )
    merged = merged.drop(columns="_merge")
    duplicates = int(merged.duplicated(subset=key_columns).sum())
    if duplicates:
        raise ValueError(f"{duplicates} duplicate rows after the label/feature join")

    orphan = feature_primary[key_columns].merge(
        labels[key_columns].drop_duplicates(), on=key_columns, how="left", indicator=True
    )
    n_orphan = int((orphan["_merge"] == "left_only").sum())
    if n_orphan:
        raise ValueError(f"{n_orphan} {track} feature rows have no matching label row")

    if not confidence.empty:
        conf_stable = stable and _has_stable_keys(confidence)
        conf_keys = key_columns if conf_stable else ["question_id", "track", "answer_index"]
        merged = merged.merge(
            confidence[conf_keys + ["value", "parse_ok", "raw_text"]].rename(
                columns={"value": "conf", "parse_ok": "conf_parse_ok", "raw_text": "conf_raw"}
            ),
            on=conf_keys,
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

    if track.upper() == "B" and expected_samples:
        sums = merged.groupby("question_id")["multiplicity"].sum()
        incomplete = sums[sums != expected_samples]
        if len(incomplete):
            raise ValueError(
                f"{len(incomplete)} Track B questions do not retain {expected_samples} draws: "
                f"{incomplete.head(10).to_dict()}"
            )
    return merged


TRACK_A_METADATA = ("relation", "stratum", "popularity", "prompt_tokens")
TRACK_B_METADATA = TRACK_A_METADATA + ("content_length",)


def metadata_features(frame: pd.DataFrame, track: str | None = None) -> pd.DataFrame:
    """Difficulty block D: relation one-hot, popularity, lengths, missingness.

    ``track="A"`` is the before-answer task, so the block is restricted to
    information available before generation: relation, stratum, popularity, and
    prompt length.  Generated-answer length and confidence missingness are
    after-generation information and are excluded.  Track B is the after-answer
    task and may use ``content_length``.

    The stated confidence score is deliberately excluded.  It belongs to the
    separate V comparator (plan section 6.1: B = D + P + H).
    """

    allow = TRACK_B_METADATA if (track or "B").upper() == "B" else TRACK_A_METADATA
    parts = []
    if "relation" in frame.columns:
        parts.append(pd.get_dummies(frame["relation"], prefix="rel").astype(float))
    if "stratum" in frame.columns:
        parts.append(pd.get_dummies(frame["stratum"], prefix="stratum").astype(float))
    for column in ("popularity", "content_length", "prompt_tokens"):
        if column in allow and column in frame.columns:
            parts.append(frame[[column]].astype(float).rename(columns={column: f"meta_{column}"}))
    if (track or "B").upper() == "B" and "conf" in frame.columns:
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
