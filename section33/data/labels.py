"""Frozen label policy for the Section 3.3 experiment.

Policy decisions (frozen before any final-test collection):

- Compare normalized text with normalized aliases from the reference source.
- Require a token-boundary match.  An alias of two characters or less must match
  the whole candidate, to avoid accidental matches inside unrelated words.
- Multiple reference answers: a match with any one of them is correct.
- Birth place: only the documented reference strings count.  No automatic
  geographic generalization or narrowing.
- Occupation synonyms: only the documented reference alias list counts.  No
  external synonym expansion.
- A candidate that adds unrelated text is not penalized if a reference answer
  appears as a token-boundary match.  Reviewers can override during adjudication.

Reviewers use ``adjudication_row`` to record decisions.  The final test uses two
independent reviews with disagreement resolution.  This module never sees probe
scores.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

POLICY_VERSION = "label_policy_v1"
_ARTICLES = re.compile(r"^(the|a|an)\s+")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, lower case, remove punctuation and articles, collapse spaces."""

    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    text = _ARTICLES.sub("", text)
    return text


def token_match(candidate_norm: str, alias_norm: str) -> bool:
    """Token-boundary containment with the short-alias rule."""

    if not candidate_norm or not alias_norm:
        return False
    candidate_tokens = candidate_norm.split()
    alias_tokens = alias_norm.split()
    if len(alias_norm) <= 2:
        return candidate_tokens == alias_tokens
    n = len(alias_tokens)
    for start in range(len(candidate_tokens) - n + 1):
        if candidate_tokens[start : start + n] == alias_tokens:
            return True
    return False


def match_candidate(candidate_text: str, references: list[str]) -> int:
    """Return one if any reference answers matches the candidate."""

    candidate = normalize(candidate_text)
    return int(any(token_match(candidate, normalize(ref)) for ref in references))


def label_policy_hash() -> str:
    payload = "|".join(
        [
            POLICY_VERSION,
            "token_boundary",
            "short_alias_full_match<=2",
            "any_reference_counts",
            "no_synonym_expansion",
            "no_geo_generalization",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def adjudication_row(
    question_id: str,
    candidate_text: str,
    reference_answers: list[str],
    reviewer: str,
) -> dict:
    """Blank review row for the two-reviewer final-test process."""

    return {
        "question_id": question_id,
        "candidate_text": candidate_text,
        "references": list(reference_answers),
        "normalized_candidate": normalize(candidate_text),
        "normalized_references": [normalize(r) for r in reference_answers],
        "reviewer": reviewer,
        "decision": None,  # 1 correct, 0 incorrect
        "note": "",
        "policy_version": POLICY_VERSION,
    }


def audit_against_adjudication(records: list[dict]) -> dict:
    """Matcher precision/recall against adjudicated rows (development audit)."""

    tp = fp = fn = tn = 0
    for row in records:
        if row.get("decision") is None:
            continue
        predicted = match_candidate(row["candidate_text"], row["references"])
        gold = int(row["decision"])
        tp += predicted == 1 and gold == 1
        fp += predicted == 1 and gold == 0
        fn += predicted == 0 and gold == 1
        tn += predicted == 0 and gold == 0
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    agreement = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "n": tp + tn + fp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "agreement": agreement,
    }
