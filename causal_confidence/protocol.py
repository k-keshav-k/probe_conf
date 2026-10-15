"""Frozen run configuration, stage gates, and protocol version.

Everything that the plan calls "frozen" lives here.  ``python -m
causal_confidence.cli setup`` writes a JSON copy with a content hash to the run
root so a cluster run can verify the design.  Do not change a frozen value
after test labels are opened; a change requires a new protocol version and
fresh evaluation data (plan section 17).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROTOCOL_VERSION = "cc_v1"

# The plan fixes the model and an explicit revision so that the first
# pretrained result stays comparable to the earlier work (plan section 6).
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

# Relations with unambiguous short answers (plan section 6).  ``occupation`` is
# excluded from the default set because the toy report and section 33 showed
# noisy occupation labels; it can be added by a protocol revision.
RELATIONS = ("father", "mother", "place_of_birth")

RELATION_TEMPLATES = {
    "father": "Who is the father of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "place_of_birth": "In what city was {subject} born?",
}

# ---------------------------------------------------------------------------
# Frozen prompts.  The prefix is shared by the answer and report branches; the
# suffixes are the only difference (plan section 5).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer each question with the shortest possible answer. Do not explain."
)
PREFIX_TEMPLATE = "{system}\n\nQuestion: {question}\n"
ANSWER_SUFFIX = "Answer:"
REPORT_SUFFIX = (
    "\n\nBefore answering, state the probability that an answer you would give "
    "now is correct, from 0.00 to 1.00 in steps of 0.05. "
    "Reply with only the number.\nProbability:"
)

# 21 constrained report values (plan section 8.1).
CONFIDENCE_GRID = tuple(round(i / 20.0, 2) for i in range(21))
REPORT_STRINGS = tuple(f"{value:.2f}" for value in CONFIDENCE_GRID)

# Six initial condition slots (plan section 7.3).  Construction, not labels.
CONDITION_SLOTS = (
    "clean",
    "target_mild",
    "target_strong",
    "restoration",
    "matched_random",
    "other_group",
)

# Main mixture weights: clean half, the other five slots one tenth each
# (plan section 11.2).
CONDITION_WEIGHTS = {
    "clean": 0.5,
    "target_mild": 0.1,
    "target_strong": 0.1,
    "restoration": 0.1,
    "matched_random": 0.1,
    "other_group": 0.1,
}

# Pair types for the contrast term (plan section 8).
PAIR_TYPES = (
    ("target_strong", "clean"),
    ("restoration", "target_strong"),
    ("matched_random", "clean"),
    ("other_group", "clean"),
)

PROMPT_VERSION = "cc_prompts_v1"
LABEL_POLICY_VERSION = "label_policy_v1"
MATCH_POLICY_VERSION = "match_policy_v1"
REPORT_GRAMMAR_VERSION = "report_grid_v1"


@dataclass(frozen=True)
class PartitionSizes:
    """Proposed pilot sizes (plan section 6).  Not a power guarantee."""

    discovery: int = 100
    validation: int = 300
    train: int = 1_000
    dev: int = 200
    test: int = 400
    transfer: int = 200

    @property
    def total(self) -> int:
        return (
            self.discovery
            + self.validation
            + self.train
            + self.dev
            + self.test
            + self.transfer
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SamplingPolicy:
    """Frozen decoding policy for outcome sampling (plan section 4)."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 32
    train_draws: int = 8
    eval_draws: int = 16


@dataclass(frozen=True)
class InterventionConfig:
    """Bounded discovery search and condition construction (plan section 7)."""

    site_fractions: tuple[float, ...] = (1 / 3, 1 / 2, 2 / 3)
    group_sizes: tuple[int, ...] = (16, 64)
    screening_attenuation: float = 0.5
    refinement_attenuations: tuple[float, ...] = (0.25, 0.75)
    max_initial_candidates: int = 6
    max_refined_candidates: int = 2
    random_replicates: int = 5
    screening_questions: int = 50
    screening_draws: int = 8
    max_screening_draws: int = 10_000
    restoration_fraction_gate: float = 0.5


@dataclass(frozen=True)
class Gates:
    """Effect-size targets.  Proposed engineering targets, not estimates."""

    # P0 (plan section 10, P0).
    p0_reproduction_mse: float = 0.001
    p0_brier_tolerance: float = 0.002
    # P2 (plan section 10, P2).
    p2_target_damage: float = 0.05
    p2_target_minus_random: float = 0.03
    p2_restoration_recovery: float = 0.5
    # P3 (plan section 10, P3).  Recorded here; used once P3 is unlocked.
    p3_brier_improvement: float = 0.005
    p3_contrast_reduction: float = 0.10
    p3_clean_guardrail: float = 0.005
    # Bootstrap.
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20261015
    confidence_level: float = 0.95


@dataclass(frozen=True)
class Protocol:
    version: str = PROTOCOL_VERSION
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    relations: tuple[str, ...] = RELATIONS
    partitions: PartitionSizes = field(default_factory=PartitionSizes)
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    gates: Gates = field(default_factory=Gates)
    seeds: tuple[int, ...] = (0, 1, 2)
    inner_seed: int = 20261015
    state_weights: dict = field(default_factory=lambda: dict(CONDITION_WEIGHTS))
    pair_lambda: float = 1.0
    pair_lambda_grid: tuple[float, ...] = (0.25, 1.0, 4.0)
    report_rank: int = 8

    def to_json_dict(self) -> dict:
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(self)

    def content_hash(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


PROTOCOL = Protocol()


def prompt_manifest() -> dict:
    """Content hashes of the frozen prompt, grammar, and policy text.

    The protocol dataclass cannot carry raw prompt strings, so their hashes are
    written next to it at setup and checked before locked evaluation.
    """

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    return {
        "system_prompt": digest(SYSTEM_PROMPT),
        "prefix_template": digest(PREFIX_TEMPLATE),
        "answer_suffix": digest(ANSWER_SUFFIX),
        "report_suffix": digest(REPORT_SUFFIX),
        "confidence_grid": digest(json.dumps(list(CONFIDENCE_GRID))),
        "relation_templates": digest(json.dumps(RELATION_TEMPLATES, sort_keys=True)),
        "prompt_version": PROMPT_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "match_policy_version": MATCH_POLICY_VERSION,
        "report_grammar_version": REPORT_GRAMMAR_VERSION,
    }


def run_paths(run_root: Path | None = None) -> dict[str, Path]:
    """Return the artifact layout for one run.

    The default root is ``runs/causal_confidence/<protocol_version>/default``.
    ``CAUSAL_RUN_DIR`` overrides it so cluster scripts can point at scratch.
    """

    if run_root is None:
        env = os.environ.get("CAUSAL_RUN_DIR")
        root = Path(env) if env else ROOT / "runs" / "causal_confidence" / PROTOCOL_VERSION / "default"
    else:
        root = Path(run_root)
    return {
        "root": root,
        "protocol": root / "protocol",
        "manifests": root / "manifests",
        "outcomes": root / "outcomes",
        "features": root / "features",
        "predictions": root / "predictions",
        "evaluation": root / "evaluation",
        "interventions": root / "interventions",
        "reporters": root / "reporters",
        "figures": root / "figures",
        "logs": root / "logs",
    }


def ensure_run_dirs(run_root: Path | None = None) -> dict[str, Path]:
    paths = run_paths(run_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_protocol(path: Path, protocol: Protocol = PROTOCOL) -> str:
    """Write the protocol JSON plus prompt hashes; return the content hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = protocol.to_json_dict()
    payload["protocol_hash"] = protocol.content_hash()
    payload["prompt_manifest"] = prompt_manifest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload["protocol_hash"]
