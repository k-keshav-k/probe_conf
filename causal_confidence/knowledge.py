"""Controlled knowledge variants: synthetic fact corpus and exposure design.

The revised experiment (after the attenuation branch) creates several versions
of the same base model with different factual knowledge, then trains one
shared confidence reporter that must recognize what each version knows without
ever seeing a variant identifier, exposure list, gold answer, or sampled
answer.

Design requirements:

- Questions are identical across versions; only the model state differs.
- Facts are invented, so the frozen base model cannot know them: training
  exposure, not prior knowledge, is what changes correctness.
- Exposure is counterbalanced: every fact is learned by exactly two of the
  four reporter-training variants, so each variant knows ~half the questions
  and every pair of variants shares one sixth.
- Splits are subject-disjoint.  A separate transfer pool (its subjects and
  facts never used for reporter training) feeds a fifth knowledge adapter used
  only at transfer evaluation.
- Every fact has one training question and several held-out paraphrases so
  knowledge verification measures generalization, not template memorization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Templates.  Attribute names are invented so no real-world relation is
# implied; values are invented names too, so the base model cannot answer.
# ---------------------------------------------------------------------------
ATTRIBUTES: tuple[str, ...] = ("mentor", "workshop", "emblem", "instrument")

TEMPLATES: dict[str, dict] = {
    "mentor": {
        "statement": "The mentor of {subject} is {value}.",
        "train_question": "Who is the mentor of {subject}?",
        "eval_questions": (
            "Who mentors {subject}?",
            "Which person is the mentor of {subject}?",
            "Who is {subject}'s mentor?",
            "Who acts as the mentor of {subject}?",
            "Which person serves as {subject}'s mentor?",
            "Who is the person mentoring {subject}?",
        ),
    },
    "workshop": {
        "statement": "{subject}'s workshop is in {value}.",
        "train_question": "Where is the workshop of {subject}?",
        "eval_questions": (
            "In which place is {subject}'s workshop?",
            "Where does {subject} keep the workshop?",
            "Which location holds {subject}'s workshop?",
            "Where can {subject}'s workshop be found?",
            "In which location is the workshop of {subject}?",
            "Where is the workshop that belongs to {subject}?",
        ),
    },
    "emblem": {
        "statement": "The emblem of {subject} is {value}.",
        "train_question": "What color is the emblem of {subject}?",
        "eval_questions": (
            "Which color marks the emblem of {subject}?",
            "What is the color of {subject}'s emblem?",
            "Which color is used for {subject}'s emblem?",
            "What color does {subject}'s emblem have?",
            "Which color belongs to the emblem of {subject}?",
            "What is the emblem color of {subject}?",
        ),
    },
    "instrument": {
        "statement": "The instrument of {subject} is {value}.",
        "train_question": "What is the instrument of {subject}?",
        "eval_questions": (
            "Which instrument belongs to {subject}?",
            "What instrument does {subject} use?",
            "Which instrument is {subject}'s?",
            "Which instrument does {subject} play?",
            "What is the instrument used by {subject}?",
            "Which instrument is used by {subject}?",
        ),
    },
}

_SYLLABLES = (
    "ka", "ro", "lin", "ves", "tar", "mo", "bel", "dru", "fen", "sar",
    "nel", "tor", "mi", "gan", "lor", "ven", "syl", "dar", "quen", "pol",
)
_VALUE_POOLS = {
    "mentor": ("Varo Kest", "Liris Mund", "Toran Vale", "Mira Belen", "Dravin Sol"),
    "workshop": ("Kaldun", "Vessport", "Oruna", "Telmara", "Brinwick"),
    "emblem": ("amber", "cobalt", "crimson", "jade", "violet"),
    "instrument": ("kestrel harp", "drum luth", "silk pipe", "glass horn", "reed organ"),
}


def _name(index: int) -> str:
    first = _SYLLABLES[index % len(_SYLLABLES)]
    second = _SYLLABLES[(index * 7 + 3) % len(_SYLLABLES)]
    return (first + second).capitalize()


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject_id: str
    subject_name: str
    attribute: str
    value: str
    statement: str
    train_question: str
    eval_questions: tuple[str, ...]
    references: tuple[str, ...]

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["eval_questions"] = list(self.eval_questions)
        payload["references"] = list(self.references)
        return payload


def build_corpus(
    subjects: int = 60, *, seed: int = 0, prefix: str = "main"
) -> list[Fact]:
    """Deterministic invented-fact corpus, four facts per subject."""

    facts: list[Fact] = []
    for subject_index in range(subjects):
        subject_id = f"{prefix}_s{subject_index:03d}"
        subject_name = _name(seed * 1000 + subject_index)
        for attribute_index, attribute in enumerate(ATTRIBUTES):
            templates = TEMPLATES[attribute]
            value = _VALUE_POOLS[attribute][
                (subject_index * 7 + attribute_index * 3 + seed) % len(_VALUE_POOLS[attribute])
            ]
            fact_id = f"{subject_id}_{attribute}"
            facts.append(
                Fact(
                    fact_id=fact_id,
                    subject_id=subject_id,
                    subject_name=subject_name,
                    attribute=attribute,
                    value=value,
                    statement=templates["statement"].format(
                        subject=subject_name, value=value
                    ),
                    train_question=templates["train_question"].format(subject=subject_name),
                    eval_questions=tuple(
                        template.format(subject=subject_name)
                        for template in templates["eval_questions"]
                    ),
                    references=(value,),
                )
            )
    return facts


# ---------------------------------------------------------------------------
# Exposure design
# ---------------------------------------------------------------------------
VARIANT_IDS: tuple[int, ...] = (1, 2, 3, 4)
TRANSFER_VARIANT_ID: int = 5
_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


def assign_exposure(facts: list[Fact]) -> dict[int, tuple[str, ...]]:
    """Balanced incomplete block design over six fact groups.

    Facts are grouped by ``index % 6``; group ``g`` is assigned to pair
    ``_PAIRS[g]``.  Each variant therefore trains on three of six groups
    (half the facts), each fact is learned by exactly two variants, and every
    variant pair shares exactly one group.
    """

    groups: dict[int, list[str]] = {group: [] for group in range(len(_PAIRS))}
    for index, fact in enumerate(facts):
        groups[index % len(_PAIRS)].append(fact.fact_id)
    exposure: dict[int, tuple[str, ...]] = {}
    for variant in VARIANT_IDS:
        ids: list[str] = []
        for group, pair in enumerate(_PAIRS):
            if variant in pair:
                ids.extend(groups[group])
        exposure[variant] = tuple(sorted(ids))
    return exposure


def exposure_summary(
    facts: list[Fact], exposure: dict[int, tuple[str, ...]]
) -> dict:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    by_variant = {
        str(variant): {
            "n_facts": len(ids),
            "n_subjects": len({facts_by_id[fact_id].subject_id for fact_id in ids}),
        }
        for variant, ids in exposure.items()
    }
    per_fact = {}
    for fact in facts:
        holders = [variant for variant, ids in exposure.items() if fact.fact_id in ids]
        per_fact[fact.fact_id] = holders
    counts = {}
    for holders in per_fact.values():
        counts[len(holders)] = counts.get(len(holders), 0) + 1
    return {
        "by_variant": by_variant,
        "facts_per_variant_count": counts,
        "balanced_pairs": all(
            len(set(exposure[first]) & set(exposure[second])) == len(facts) // 6
            for first, second in _PAIRS
        ),
    }


def split_by_subject(
    facts: list[Fact],
    *,
    train_subjects: int = 36,
    dev_subjects: int = 12,
) -> dict[str, list[Fact]]:
    """Subject-disjoint reporter splits inside one corpus."""

    subjects = list(dict.fromkeys(fact.subject_id for fact in facts))
    if train_subjects + dev_subjects > len(subjects):
        raise ValueError("requested splits exceed the subject pool")
    train_ids = set(subjects[:train_subjects])
    dev_ids = set(subjects[train_subjects : train_subjects + dev_subjects])
    splits = {"train": [], "dev": [], "test": []}
    for fact in facts:
        if fact.subject_id in train_ids:
            splits["train"].append(fact)
        elif fact.subject_id in dev_ids:
            splits["dev"].append(fact)
        else:
            splits["test"].append(fact)
    return splits


def corpus_manifest(
    facts: list[Fact],
    exposure: dict[int, tuple[str, ...]],
    splits: dict[str, list[Fact]],
    transfer: list[Fact],
    *,
    seed: int,
) -> dict:
    payload = {
        "seed": seed,
        "n_facts": len(facts),
        "n_transfer_facts": len(transfer),
        "exposure": {str(k): list(v) for k, v in exposure.items()},
        "splits": {name: [fact.fact_id for fact in items] for name, items in splits.items()},
        "transfer_manifest": [fact.fact_id for fact in transfer],
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def write_corpus_manifest(path: str | Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload["content_hash"]
