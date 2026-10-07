"""Shared configuration and utilities for the probe_conf pilot."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (DATA_DIR, RESULTS_DIR, RAW_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SMOKE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# PopQA person relations. NOTE: PopQA has no "date of birth"; "place of birth"
# is the substitute for the pilot (see PLAN.md section 7).
PERSON_RELATIONS = ["father", "mother", "place of birth", "occupation"]

SEED = 0
N_QUESTIONS = 1000
N_PER_RELATION_BIN = 60  # stratified by s_pop quintiles within relation
K_SAMPLES = 30
MAX_NEW_TOKENS = 16
TEMPERATURE = 1.0
TOP_P = 1.0

# Feature extraction
RAW_LAYER_STRIDE = 1  # store raw MLP vectors for every layer at the answer pos
N_COMPETITORS = 50

SYSTEM_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer each question with the shortest possible answer. Do not explain."
)

CONFIDENCE_PROMPT = (
    "{question}\n\n"
    "If you had to state the probability that your answer to this question is "
    "correct as a number from 1 to 100, what number would you give? "
    "Answer with just the number."
)


# ---------------------------------------------------------------------------
# Text normalization and answer matching
# ---------------------------------------------------------------------------
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _ARTICLE_RE.sub("", text)
    return text


def matches_any_alias(generation: str, aliases: list[str]) -> bool:
    """True if any alias appears as a token sequence in the generation.

    Short aliases (<= 2 chars) require an exact match to avoid false positives
    (e.g. "US" inside "because").
    """
    gen = normalize_answer(generation)
    if not gen:
        return False
    gen_tokens = gen.split()
    for alias in aliases:
        a = normalize_answer(alias)
        if not a:
            continue
        a_tokens = a.split()
        if len(a) <= 2:
            if gen_tokens == a_tokens:
                return True
            continue
        # contiguous token subsequence match
        n = len(a_tokens)
        for i in range(len(gen_tokens) - n + 1):
            if gen_tokens[i : i + n] == a_tokens:
                return True
        if a in gen:  # Historical substring fallback; can accept partial words.
            return True
    return False


def first_line(text: str) -> str:
    """Take the first non-empty line and strip common chat artifacts."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------
@dataclass
class Question:
    id: int
    subj: str
    subj_id: int
    prop: str
    obj: str
    question: str
    aliases: list[str] = field(default_factory=list)
    s_pop: int = 0
    o_pop: int = 0
    prompt: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or (DATA_DIR / "popqa_person.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            out.append(Question(**json.loads(line)))
    return out


def entropy_from_probs(p) -> float:
    """Shannon entropy (nats) of a 1-D probability vector (numpy or torch)."""
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def gini(x) -> float:
    """Gini coefficient of a nonnegative 1-D array."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def safe_nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6):
    import numpy as np

    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=nan, posinf=posinf, neginf=neginf)
