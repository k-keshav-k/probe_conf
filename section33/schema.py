"""Row schemas for the Section 3.3 run.

Two storage interfaces are kept strictly apart:

- **Labels** (``labels/``): question IDs, candidate answers, reference answers,
  and correctness labels.  These files drive evaluation.
- **Features** (``features/``): token IDs, positions, seeds, and reference-free
  feature values.  These files never contain a reference answer or a label.

The plan requires this separation (section 3.3: "No generated token, reference
token, answer alias, object popularity, or correctness label can enter the
feature extractor").  The dataclasses below make the boundary explicit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnswerRecord:
    """One generated candidate answer and its label (labels interface)."""

    model: str
    revision: str
    subject_id: str
    question_id: str
    split: str
    track: str  # "A" or "B"
    answer_index: int
    generation_seed: int
    candidate_text: str
    candidate_key: str
    label: Optional[int]
    label_policy_version: str
    reference_answers: list[str] = field(default_factory=list)
    empty: bool = False
    truncated: bool = False
    refusal: bool = False
    exception: Optional[str] = None
    multiplicity: int = 1
    relation: str = ""
    popularity: Optional[float] = None
    stratum: str = ""

    def to_row(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class FeatureRow:
    """Reference-free features for one candidate and one position.

    There is deliberately no field for the reference answer or the label.  If a
    function needs them, it is a label operation and must not write here.
    """

    model: str
    revision: str
    subject_id: str
    question_id: str
    split: str
    track: str
    answer_index: int
    generation_seed: int
    position: str  # "last_prompt", "answer_last_content", "last_subject", "content_mean"
    features: dict[str, float]
    prompt_token_hash: str
    candidate_token_ids: list[int]
    candidate_key: str
    content_length: int
    feature_spec_version: str
    multiplicity: int = 1

    def to_row(self) -> dict:
        row = dataclasses.asdict(self)
        features = row.pop("features")
        row.update({f"f__{k}": v for k, v in features.items()})
        return row


@dataclass
class PredictionRow:
    """A probe score for one (method, model, candidate) tuple."""

    method: str
    probe_id: str
    model: str
    revision: str
    subject_id: str
    question_id: str
    split: str
    track: str
    answer_index: int
    generation_seed: int
    position: str
    target: int
    raw_score: float
    calibrated_score: Optional[float]
    fold: int
    seed: int
    label_version_hash: str
    multiplicity: int = 1

    def to_row(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ConfidenceRecord:
    """One parsed confidence report (labels interface; text kept for audit)."""

    model: str
    revision: str
    subject_id: str
    question_id: str
    split: str
    track: str
    answer_index: int
    candidate_key: str
    generation_seed: int
    prompt_version: str
    raw_text: str
    value: Optional[float]
    parse_ok: bool

    def to_row(self) -> dict:
        return dataclasses.asdict(self)


REQUIRED_PREDICTION_FIELDS = (
    "model",
    "revision",
    "subject_id",
    "question_id",
    "split",
    "track",
    "answer_index",
    "generation_seed",
    "position",
    "target",
    "raw_score",
    "calibrated_score",
    "probe_id",
    "label_version_hash",
)
