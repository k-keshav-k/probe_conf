"""Frozen protocol settings for the Section 3.3 confirmatory experiment.

Everything that the analysis plan calls "frozen" lives here.  A JSON copy with a
content hash is written to ``runs/section33_v1/protocol/protocol.json`` by
``python -m section33.cli setup`` so that the cluster run can verify the design.

Do not change a frozen value after final-test labels are opened.  Development-
only choices are marked with ``dev_`` or are selected by the search loops.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# ``cluster/env.sh`` exports RUN_DIR; honour it so a manual command without an
# explicit ``--run`` uses the same run root as the sbatch scripts.  Local use
# without RUN_DIR falls back to the repository copy.
RUN_ROOT = Path(os.environ.get("RUN_DIR", ROOT / "runs" / "section33_v1"))

# ---------------------------------------------------------------------------
# Study definition
# ---------------------------------------------------------------------------
RELATIONS = ("father", "mother", "place_of_birth", "occupation")
RELATION_QIDS = {
    "father": "P22",
    "mother": "P25",
    "place_of_birth": "P19",
    "occupation": "P106",
}
# Frozen question phrasing for the fresh Wikidata cohort (mirrors the few-shot
# templates).  ``wikidata.fetch_cohort`` fills these in before the cohort job.
RELATION_QUESTIONS = {
    "father": "Who is the father of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "place_of_birth": "In what city was {subject} born?",
    "occupation": "What is {subject}'s occupation?",
}
POPULARITY_BINS = 5
MISSING_POPULARITY_STRATUM = "missing"


@dataclass(frozen=True)
class Margins:
    """Smallest effects worth claiming (plan section 10.1)."""

    main_auroc: float = 0.020
    main_brier: float = 0.005
    within_question_auroc: float = 0.030
    calibration_epsilon: float = 0.010
    equivalence_auroc: float = 0.020


@dataclass(frozen=True)
class SplitSizes:
    train: int = 2_000
    dev: int = 800
    calibration: int = 800
    test: int = 4_000
    test_cap: int = 6_000

    @property
    def max_cohort(self) -> int:
        return self.train + self.dev + self.calibration + self.test_cap


@dataclass(frozen=True)
class Protocol:
    version: str = "section33_v1"
    models: tuple[str, ...] = (
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    )
    primary_model: str = "Qwen/Qwen2.5-7B-Instruct"
    replication_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    margins: Margins = field(default_factory=Margins)
    splits: SplitSizes = field(default_factory=SplitSizes)

    # Track definitions.
    track_a_samples: int = 1
    track_b_samples: int = 8
    content_token_limit: int = 32
    temperature: float = 1.0
    top_p: float = 1.0

    # Probe fitting (plan section 6.2).
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    inner_folds: int = 3
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    max_configs_per_side: int = 24
    pca_budget: int = 512
    pca_hidden_half: int = 256
    pca_mlp_half: int = 256
    nonlinear_hidden: int = 128
    nonlinear_lr: tuple[float, ...] = (1e-4, 1e-3)
    nonlinear_wd: tuple[float, ...] = (1e-4, 1e-3)
    nonlinear_epochs: int = 100
    nonlinear_patience: int = 10

    # Calibration (plan section 6.3).
    calibration_methods: tuple[str, ...] = ("identity", "platt", "isotonic")
    coverage_levels: tuple[float, ...] = (0.25, 0.50, 0.75)
    target_error_rate: float = 0.10

    # Statistics (plan section 9).
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20261007
    # Frozen family for simultaneous secondary intervals (Bonferroni across the
    # family; the four AUROC p-values additionally use Holm).
    secondary_family: tuple[str, ...] = (
        "replication_hidden_plus_mlp_minus_hidden",
        "mlp_minus_hidden",
        "augmented_minus_confidence",
        "augmented_minus_probability",
        "within_question",
        "mlp_minus_hidden_equivalence",
    )

    # Interventions (plan section 8.3).
    intervention_top_k: tuple[int, ...] = (1, 5, 20)
    intervention_attenuation: tuple[float, ...] = (0.0, 0.5)
    intervention_exponents: tuple[float, ...] = (0.5, 1.0, 2.0)
    intervention_caliper: float = 0.10

    # Feature-family registry names (implemented in features.families).
    families: tuple[str, ...] = (
        "w2_literal",
        "gate_selection",
        "contribution_concentration",
        "output_direction",
        "pilot12",
    )
    pilot_reference_free: int = 12
    pilot_reference_dependent: tuple[str, ...] = (
        "cos_sel_agold",
        "proj_gold_norm",
        "mlp_margin",
        "ent_cand",
        "logit_mlp_gold",
    )

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

# ---------------------------------------------------------------------------
# Frozen prompts (plan sections 5.2 and 5.3)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer each question with the shortest possible answer. Do not explain."
)
FEWSHOT: tuple[tuple[str, str], ...] = (
    ("Who is the father of George W. Bush?", "George H. W. Bush"),
    ("Who is the mother of Prince William?", "Diana, Princess of Wales"),
    ("In what city was Albert Einstein born?", "Ulm"),
    ("What is Barack Obama's occupation?", "politician"),
)
STOP_STRINGS = ("\n",)

CONFIDENCE_TRACK_A = (
    "{question}\n\n"
    "If you had to state the probability that your next short answer to this "
    "question is correct, as an integer from 0 to 100, what integer would you give? "
    "Answer with only the integer."
)

CONFIDENCE_TRACK_B = (
    "{question}\n\n"
    "A candidate answer is: {candidate}\n\n"
    "If you had to state the probability that this candidate answer is correct, "
    "as an integer from 0 to 100, what integer would you give? "
    "Answer with only the integer."
)

# Backup names kept for consistency with the pilot templates.
PROMPT_VERSION = "section33_prompts_v1"
LABEL_POLICY_VERSION = "label_policy_v1"
MATCH_POLICY_VERSION = "match_policy_v1"


def prompt_manifest() -> dict:
    """Content hashes of the frozen prompt and policy text.

    The protocol dataclass cannot carry the prompt strings, so their hashes are
    written next to it at setup and verified before evaluation (review F12).
    """

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    return {
        "system_prompt": digest(SYSTEM_PROMPT),
        "fewshot": digest(json.dumps(FEWSHOT)),
        "stop_strings": digest(json.dumps(STOP_STRINGS)),
        "confidence_track_a": digest(CONFIDENCE_TRACK_A),
        "confidence_track_b": digest(CONFIDENCE_TRACK_B),
        "relation_questions": digest(json.dumps(RELATION_QUESTIONS)),
        "prompt_version": PROMPT_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "match_policy_version": MATCH_POLICY_VERSION,
    }


def run_paths(run_root: Path | None = None) -> dict[str, Path]:
    """Return the artefact layout required by plan section 12."""

    root = Path(run_root) if run_root else RUN_ROOT
    return {
        "root": root,
        "protocol": root / "protocol",
        "manifests": root / "manifests",
        "labels": root / "labels",
        "features": root / "features",
        "probes": root / "probes",
        "predictions": root / "predictions",
        "evaluation": root / "evaluation",
        "interventions": root / "interventions",
        "report": root / "report",
        "logs": root / "logs",
    }


def write_protocol(path: Path, protocol: Protocol = PROTOCOL) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = protocol.to_json_dict()
    payload["protocol_hash"] = protocol.content_hash()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload["protocol_hash"]
