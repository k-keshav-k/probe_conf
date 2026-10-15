"""Subject splits, audited answers, relation definitions, and duplicates.

All partitions are subject-disjoint: every relation of a subject stays in one
partition.  The manifest records content hashes, per-relation counts, and the
achieved sizes so a reduced pilot is visible rather than silent (plan section
6).  Reference and alias audits run before any labels are collected.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .protocol import PROTOCOL, Protocol

# PopQA property names -> protocol relation keys.
POPQA_RELATIONS = {
    "father": "father",
    "mother": "mother",
    "place of birth": "place_of_birth",
    "place_of_birth": "place_of_birth",
}


@dataclass(frozen=True)
class Question:
    question_id: str
    subject_id: str
    subject_name: str
    relation: str
    question_text: str
    references: tuple[str, ...]
    popularity: float | None = None
    source: str = "unknown"

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["references"] = list(self.references)
        return payload


def _question_key(row: dict) -> str:
    return hashlib.sha256(
        f"{row['subject_id']}|{row['relation']}|{row['question_text']}".encode()
    ).hexdigest()[:16]


def load_cohort(path: str | Path) -> list[Question]:
    """Load the section 33 cohort parquet (subject/relation/reference rows)."""

    import pandas as pd

    frame = pd.read_parquet(path)
    required = {"subject_id", "relation", "question", "reference_answers"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cohort is missing columns: {sorted(missing)}")
    rows: list[Question] = []
    for record in frame.to_dict("records"):
        relation = str(record["relation"])
        if relation not in PROTOCOL.relations:
            continue
        references = record["reference_answers"]
        if isinstance(references, str):
            references = json.loads(references)
        rows.append(
            Question(
                question_id=str(
                    record.get("question_id", _question_key(record))
                ),
                subject_id=str(record["subject_id"]),
                subject_name=str(record.get("subject_name", record["subject_id"])),
                relation=relation,
                question_text=str(record["question"]),
                references=tuple(str(ref) for ref in references),
                popularity=(
                    float(record["popularity"])
                    if record.get("popularity") is not None
                    else None
                ),
                source="section33_cohort",
            )
        )
    return rows


def load_popqa(path: str | Path) -> list[Question]:
    """Load PopQA-style JSONL with subject/object aliases.

    Both the raw PopQA JSONL (``obj``/``aliases``) and the section 33 source
    parquet (``possible_answers``) are accepted.
    """

    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        records = pd.read_parquet(path).to_dict("records")
    else:
        records = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    rows: list[Question] = []
    for record in records:
        prop = str(record.get("prop", record.get("relation", ""))).strip()
        relation = POPQA_RELATIONS.get(prop)
        if relation is None:
            continue
        references: list[str] = []
        for key in ("aliases", "o_aliases", "possible_answers"):
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [value]
            references.extend(str(item) for item in value)
        if record.get("obj") is not None:
            references.append(str(record["obj"]))
        subject_id = str(record.get("subj_id", record.get("subject_id")))
        question_text = str(record.get("question", "")).strip()
        if not question_text:
            continue
        rows.append(
            Question(
                question_id=str(record.get("id", record.get("question_id", _question_key(record)))),
                subject_id=subject_id,
                subject_name=str(record.get("subj", record.get("subject_name", subject_id))),
                relation=relation,
                question_text=question_text,
                references=tuple(dict.fromkeys(ref for ref in references if ref)),
                popularity=(
                    float(record["s_pop"])
                    if record.get("s_pop") is not None
                    else None
                ),
                source="popqa",
            )
        )
    return rows


def dedupe_questions(
    rows: list[Question], source_priority: tuple[str, ...] = ("section33_cohort", "popqa")
) -> tuple[list[Question], dict]:
    """Keep one question per (subject, relation); merge references.

    Duplicate questions for the same subject and relation are a known source of
    inflated counts.  The kept row prefers the earliest source in
    ``source_priority``; references from all duplicates are merged so the
    matcher sees the union of audited aliases.
    """

    priority = {name: index for index, name in enumerate(source_priority)}
    groups: dict[tuple[str, str], list[Question]] = {}
    for row in rows:
        groups.setdefault((row.subject_id, row.relation), []).append(row)
    kept: list[Question] = []
    duplicates = 0
    merged_aliases = 0
    for (subject_id, relation), group in sorted(groups.items()):
        duplicates += len(group) - 1
        group.sort(key=lambda row: (priority.get(row.source, 99), row.question_id))
        first = group[0]
        references: list[str] = []
        for item in group:
            for reference in item.references:
                if reference not in references:
                    references.append(reference)
        if len(references) != len(first.references):
            merged_aliases += 1
        kept.append(
            Question(
                question_id=first.question_id,
                subject_id=subject_id,
                subject_name=first.subject_name,
                relation=relation,
                question_text=first.question_text,
                references=tuple(references),
                popularity=first.popularity,
                source=first.source,
            )
        )
    report = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "duplicate_rows": duplicates,
        "groups_with_merged_aliases": merged_aliases,
    }
    return kept, report


def audit_references(rows: list[Question]) -> dict:
    """Reference and alias audit required before label collection (plan P1)."""

    missing = [row.question_id for row in rows if not row.references]
    blank = [
        row.question_id
        for row in rows
        if any(not str(ref).strip() for ref in row.references)
    ]
    multi = sum(len(row.references) > 1 for row in rows)
    by_relation: dict[str, int] = {}
    for row in rows:
        by_relation[row.relation] = by_relation.get(row.relation, 0) + 1
    subjects = {row.subject_id for row in rows}
    duplicate_questions = len(rows) - len({row.question_text for row in rows})
    return {
        "n_questions": len(rows),
        "n_subjects": len(subjects),
        "by_relation": dict(sorted(by_relation.items())),
        "missing_references": missing,
        "blank_references": blank,
        "questions_with_multiple_references": multi,
        "duplicate_question_texts": duplicate_questions,
        "passed": not missing and not blank,
    }


def _allocate_by_relation(
    rows: list[Question],
    sizes: dict[str, int],
    *,
    seed: int,
    transfer_relation: str | None,
) -> dict[str, list[Question]]:
    """Assign whole subjects to partitions, proportional by relation.

    Subjects are shuffled with ``seed``; each relation is allocated separately
    with largest-remainder rounding so every partition receives roughly its
    requested share of each relation.  A subject never appears twice.  When a
    ``transfer_relation`` is designated, all of its subjects go to the
    transfer partition and the other partitions share the rest.
    """

    generator = random.Random(seed)
    subjects_by_relation: dict[str, list[str]] = {}
    for row in rows:
        bucket = subjects_by_relation.setdefault(row.relation, [])
        if row.subject_id not in bucket:
            bucket.append(row.subject_id)
    designated_transfer = transfer_relation is not None
    assigned: dict[str, set[str]] = {name: set() for name in sizes}
    allocated: set[str] = set()
    for relation, subject_ids in sorted(subjects_by_relation.items()):
        subject_ids = [subject for subject in subject_ids if subject not in allocated]
        generator.shuffle(subject_ids)
        if not subject_ids:
            continue
        if designated_transfer and transfer_relation == relation:
            assigned["transfer"].update(subject_ids)
            allocated.update(subject_ids)
            continue
        requests = {
            name: float(size)
            for name, size in sizes.items()
            if not (designated_transfer and name == "transfer")
        }
        requested = sum(requests.values()) or 1.0
        total = float(len(subject_ids))
        shares = {name: total * size / requested for name, size in requests.items()}
        counts = {name: int(math.floor(share)) for name, share in shares.items()}
        remainder = int(total) - sum(counts.values())
        order = sorted(
            shares, key=lambda name: (shares[name] - counts[name]), reverse=True
        )
        for name in order[: max(0, remainder)]:
            counts[name] += 1
        cursor = 0
        for name in sizes:
            if designated_transfer and name == "transfer":
                continue
            take = min(counts.get(name, 0), max(0, len(subject_ids) - cursor))
            assigned[name].update(subject_ids[cursor : cursor + take])
            cursor += take
        # Any remainder (pool smaller than requested) goes to training first.
        if cursor < len(subject_ids):
            assigned["train"].update(subject_ids[cursor:])
        for name in sizes:
            allocated.update(assigned[name])
    partitions: dict[str, list[Question]] = {name: [] for name in sizes}
    for row in rows:
        for name in sizes:
            if row.subject_id in assigned[name]:
                partitions[name].append(row)
                break
    for name in partitions:
        partitions[name].sort(key=lambda row: (row.relation, row.subject_id, row.question_id))
    return partitions


def build_partitions(
    rows: list[Question],
    sizes: dict[str, int] | None = None,
    *,
    seed: int = PROTOCOL.inner_seed,
    allow_reduced: bool = False,
    transfer_relation: str | None = None,
) -> tuple[dict[str, list[Question]], dict]:
    """Return subject-disjoint partitions and an allocation report.

    ``allow_reduced`` permits a pool smaller than the requested pilot sizes;
    the achieved counts are recorded and the manifest is marked ``reduced``.
    Without it, a shortfall raises so a silent shrink cannot happen.
    """

    if sizes is None:
        sizes = PROTOCOL.partitions.as_dict()
    if transfer_relation is not None and transfer_relation not in PROTOCOL.relations:
        raise ValueError(f"transfer_relation {transfer_relation!r} not in {PROTOCOL.relations}")
    partitions = _allocate_by_relation(
        rows, sizes, seed=seed, transfer_relation=transfer_relation
    )
    achieved = {name: len(items) for name, items in partitions.items()}
    shortfall = {
        name: sizes[name] - achieved[name]
        for name in sizes
        if achieved[name] < sizes[name]
    }
    if shortfall and not allow_reduced:
        raise ValueError(
            "question pool cannot satisfy the frozen partition sizes; "
            f"shortfalls {shortfall}.  Pass allow_reduced to record a reduced pilot.  "
            "Do not silently shrink the study."
        )
    overlap = check_partition_isolation(partitions)
    report = {
        "requested": sizes,
        "achieved": achieved,
        "shortfall": shortfall,
        "reduced": bool(shortfall),
        "subject_overlap": overlap,
        "transfer_relation": transfer_relation,
        "seed": seed,
    }
    if overlap:
        raise ValueError(f"partition subject overlap: {overlap}")
    return partitions, report


def check_partition_isolation(partitions: dict[str, list[Question]]) -> dict[str, list[str]]:
    """Return subjects that appear in more than one partition."""

    seen: dict[str, str] = {}
    overlap: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        for row in rows:
            previous = seen.get(row.subject_id)
            if previous is not None and previous != name:
                overlap.setdefault(row.subject_id, set()).update({previous, name})
            seen[row.subject_id] = name
    return {subject: sorted(names) for subject, names in sorted(overlap.items())}


def partition_summary(partitions: dict[str, list[Question]]) -> dict:
    summary = {}
    for name, rows in partitions.items():
        by_relation: dict[str, int] = {}
        for row in rows:
            by_relation[row.relation] = by_relation.get(row.relation, 0) + 1
        summary[name] = {
            "n_questions": len(rows),
            "n_subjects": len({row.subject_id for row in rows}),
            "by_relation": dict(sorted(by_relation.items())),
        }
    return summary


def manifest_hash(partitions: dict[str, list[Question]]) -> str:
    payload = {
        name: [
            {
                "q": row.question_id,
                "s": row.subject_id,
                "r": row.relation,
                "text": row.question_text,
                "refs": list(row.references),
            }
            for row in rows
        ]
        for name, rows in sorted(partitions.items())
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_manifest(
    partitions: dict[str, list[Question]],
    path: str | Path,
    *,
    allocation: dict | None = None,
    sources: list[str] | None = None,
    protocol: Protocol = PROTOCOL,
) -> dict:
    """Write the immutable split manifest and return it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": protocol.version,
        "protocol_hash": protocol.content_hash(),
        "manifest_hash": manifest_hash(partitions),
        "sources": sources or [],
        "allocation": allocation or {},
        "partitions": {
            name: [row.to_json() for row in rows]
            for name, rows in partitions.items()
        },
        "summary": partition_summary(partitions),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_manifest(path: str | Path) -> dict[str, list[Question]]:
    payload = json.loads(Path(path).read_text())
    return {
        name: [
            Question(
                question_id=str(row["question_id"]),
                subject_id=str(row["subject_id"]),
                subject_name=str(row["subject_name"]),
                relation=str(row["relation"]),
                question_text=str(row["question_text"]),
                references=tuple(row["references"]),
                popularity=row.get("popularity"),
                source=str(row.get("source", "manifest")),
            )
            for row in rows
        ]
        for name, rows in payload["partitions"].items()
    }


def leakage_scan(texts: list[str], forbidden: list[str]) -> dict:
    """Check that forbidden strings (gold answers, condition names) are absent.

    Used on serialized reporter inputs.  Matching is case-insensitive and
    token-free; the caller decides what the forbidden list contains.
    """

    lowered = [(index, text.lower()) for index, text in enumerate(texts)]
    hits = []
    for needle in forbidden:
        needle_lower = needle.lower()
        for index, text in lowered:
            if needle_lower and needle_lower in text:
                hits.append({"index": index, "needle": needle})
    return {"n_texts": len(texts), "n_forbidden": len(forbidden), "hits": hits, "clean": not hits}
