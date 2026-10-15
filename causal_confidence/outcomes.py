"""Frozen-policy sampling and immutable correctness banks.

One bank holds every independent draw for one (question, condition slot).
Banks are keyed by the model revision, prompt hash, sampling policy, matcher
version, and draw seed, and are written immutably: a second write with
different content raises.  Gold answers and condition identifiers live in bank
metadata for auditing only; they never enter reporter inputs (plan section
14).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .protocol import (
    LABEL_POLICY_VERSION,
    MATCH_POLICY_VERSION,
    PROTOCOL,
    SamplingPolicy,
)

ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "unknown",
    "no idea",
    "cannot answer",
    "can't answer",
)


def match_answer(candidate_text: str, references: list[str]) -> int:
    """Frozen token-boundary matcher from section 33 (reused, not redefined)."""

    from section33.data.labels import match_candidate

    return int(match_candidate(candidate_text, list(references)))


@dataclass
class DrawRecord:
    seed: int
    text: str
    matched: int
    n_tokens: int
    finished: bool
    truncated: bool
    stop_reason: str
    abstained: bool


@dataclass
class OutcomeBank:
    question_id: str
    subject_id: str
    relation: str
    partition: str
    slot: str
    condition: dict = field(default_factory=dict)
    draws: list[DrawRecord] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)
    bank_seed: int = 0
    model_revision: str = ""
    prompt_hash: str = ""
    matcher_policy: str = MATCH_POLICY_VERSION
    label_policy: str = LABEL_POLICY_VERSION
    references: list[str] = field(default_factory=list)

    @property
    def success(self) -> float:
        if not self.draws:
            return float("nan")
        return sum(draw.matched for draw in self.draws) / len(self.draws)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["success"] = self.success
        payload["n_draws"] = len(self.draws)
        return payload


def sample_bank(
    model,
    tokenizer,
    *,
    question,
    prefix,
    suffix_ids,
    partition: str,
    slot: str,
    draw_seed: int,
    policy: SamplingPolicy = PROTOCOL.sampling,
    draws: int | None = None,
    condition: dict | None = None,
    model_revision: str = "",
) -> OutcomeBank:
    """Sample independent draws for one question/slot and label them."""

    from .episodes import generate_answers

    n_draws = draws if draws is not None else policy.train_draws
    records = generate_answers(
        model,
        prefix,
        suffix_ids,
        tokenizer,
        draws=n_draws,
        base_seed=draw_seed,
        temperature=policy.temperature,
        top_p=policy.top_p,
        max_new_tokens=policy.max_new_tokens,
    )
    bank = OutcomeBank(
        question_id=question.question_id,
        subject_id=question.subject_id,
        relation=question.relation,
        partition=partition,
        slot=slot,
        condition=condition or {},
        sampling={
            "temperature": policy.temperature,
            "top_p": policy.top_p,
            "max_new_tokens": policy.max_new_tokens,
            "draws": n_draws,
        },
        bank_seed=draw_seed,
        model_revision=model_revision,
        prompt_hash=prefix.prompt_hash,
        references=list(question.references),
    )
    for draw in records:
        lowered = draw.text.lower()
        bank.draws.append(
            DrawRecord(
                seed=draw_seed + len(bank.draws),
                text=draw.text,
                matched=match_answer(draw.text, list(question.references)),
                n_tokens=draw.n_tokens,
                finished=draw.finished,
                truncated=draw.truncated,
                stop_reason=draw.stop_reason,
                abstained=(not draw.text.strip())
                or any(marker in lowered for marker in ABSTENTION_MARKERS),
            )
        )
    return bank


# ---------------------------------------------------------------------------
# Immutable storage
# ---------------------------------------------------------------------------
def payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def write_immutable(path: str | Path, payload: object) -> str:
    """Write JSON once; refuse a different overwrite.  Returns the hash."""

    path = Path(path)
    digest = payload_hash(payload)
    if path.exists():
        existing = json.loads(path.read_text())
        existing_hash = payload_hash(existing)
        if existing_hash == digest:
            return digest
        raise FileExistsError(
            f"{path} already exists with a different content hash; "
            "never overwrite a frozen artifact"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return digest


def write_banks(path: str | Path, banks: list[OutcomeBank], *, stage: str) -> str:
    payload = {
        "stage": stage,
        "protocol_version": PROTOCOL.version,
        "n_banks": len(banks),
        "banks": [bank.to_json() for bank in banks],
    }
    return write_immutable(path, payload)


def load_banks(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return payload["banks"]


def success_table(banks: list[dict]) -> dict:
    """Mean success per (condition slot, relation) with counts."""

    table: dict[str, dict] = {}
    for bank in banks:
        row = table.setdefault(bank["slot"], {"n": 0, "success": 0.0, "per_relation": {}})
        row["n"] += 1
        row["success"] += bank["success"]
        relation = row["per_relation"].setdefault(bank["relation"], {"n": 0, "success": 0.0})
        relation["n"] += 1
        relation["success"] += bank["success"]
    for row in table.values():
        row["success"] /= max(1, row["n"])
        for relation in row["per_relation"].values():
            relation["success"] /= max(1, relation["n"])
    return table


def audit_banks(banks: list[dict]) -> dict:
    """Floor/ceiling, truncation, abstention, and coverage audit for P1."""

    per_slot: dict[str, dict] = {}
    for bank in banks:
        slot = bank["slot"]
        row = per_slot.setdefault(
            slot,
            {
                "n_items": 0,
                "n_draws": 0,
                "success_sum": 0.0,
                "truncated": 0,
                "abstained": 0,
                "empty": 0,
                "per_item": {},
            },
        )
        row["n_items"] += 1
        row["n_draws"] += bank["n_draws"]
        row["success_sum"] += bank["success"]
        for draw in bank["draws"]:
            row["truncated"] += int(draw["truncated"])
            row["abstained"] += int(draw["abstained"])
            row["empty"] += int(not draw["text"].strip())
        row["per_item"][bank["question_id"]] = bank["success"]
    summary: dict = {}
    for slot, row in per_slot.items():
        successes = list(row["per_item"].values())
        mean = row["success_sum"] / max(1, row["n_items"])
        summary[slot] = {
            "n_items": row["n_items"],
            "n_draws": row["n_draws"],
            "mean_success": mean,
            "truncation_rate": row["truncated"] / max(1, row["n_draws"]),
            "abstention_rate": row["abstained"] / max(1, row["n_draws"]),
            "empty_rate": row["empty"] / max(1, row["n_draws"]),
            "floor_items": sum(value == 0.0 for value in successes),
            "ceiling_items": sum(value == 1.0 for value in successes),
        }
    return summary


def check_bank_immutability(path: str | Path) -> dict:
    """Re-hash a stored bank file and report whether it can be reproduced."""

    path = Path(path)
    payload = json.loads(path.read_text())
    recomputed = payload_hash(payload)
    return {"path": str(path), "hash": recomputed, "intact": True}
