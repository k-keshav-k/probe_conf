"""Prepare the PopQA person-subject subset for the pilot.

Downloads PopQA from the HuggingFace hub, filters to person relations,
stratifies a sample by subject popularity (s_pop) within each relation,
and renders the QA prompt for each question.

Usage:
    python scripts/prepare_data.py [--n 1000] [--per-bin 60] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    DATA_DIR,
    N_PER_RELATION_BIN,
    N_QUESTIONS,
    PERSON_RELATIONS,
    SEED,
    SYSTEM_PROMPT,
    Question,
)


def render_prompt(question: str, tokenizer=None) -> str:
    """Render the chat prompt. The tokenizer is optional (falls back to plain)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"System: {SYSTEM_PROMPT}\nUser: {question}\nAssistant:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUESTIONS, help="target total questions")
    ap.add_argument("--per-bin", type=int, default=N_PER_RELATION_BIN)
    ap.add_argument("--mix", choices=["head", "uniform"], default="head")
    ap.add_argument("--force", action="store_true", help="re-download / overwrite output")
    args = ap.parse_args()

    out_path = DATA_DIR / "popqa_person.jsonl"
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; use --force to rebuild.")
        return

    from datasets import load_dataset

    print("Loading PopQA ...")
    ds = load_dataset("akariasai/PopQA", split="test")
    print(f"  total questions: {len(ds)}")
    props = ds.unique("prop")
    print(f"  available relations: {sorted(props)}")
    missing = [r for r in PERSON_RELATIONS if r not in props]
    if missing:
        raise SystemExit(f"Relations not present in PopQA: {missing}")

    rows = [row for row in ds if row["prop"] in PERSON_RELATIONS]
    print(f"  person-relation questions: {len(rows)}")

    # Parse aliases / possible answers robustly across datasets versions.
    def get_aliases(row) -> list[str]:
        for key in ("possible_answers", "o_aliases"):
            val = row.get(key)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    val = [val]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                out = [str(v) for v in val if str(v).strip()]
                if out:
                    return out
        return [str(row["obj"])]

    # Stratify by popularity. Default "head": popularity quintiles weighted
    # 10/15/20/25/30% (least -> most popular) so the model has enough signal to
    # produce positive labels while keeping difficulty spread. "uniform" gives
    # each of 50 log-s_pop bins equal weight.
    logs = np.log10(np.array([float(r["s_pop"]) for r in rows]) + 1.0)

    def bin_ids(mix: str) -> np.ndarray:
        if mix == "uniform":
            n_bins = 50
            edges = np.quantile(logs, np.linspace(0, 1, n_bins + 1))
            edges[-1] += 1e-9
            return np.clip(
                np.searchsorted(edges, logs, side="right") - 1, 0, n_bins - 1
            )
        edges = np.quantile(logs, [0.2, 0.4, 0.6, 0.8])
        return np.searchsorted(edges, logs, side="right").clip(0, 4)

    bins = bin_ids(args.mix)
    quint_weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])

    rng = np.random.default_rng(SEED)
    by_rel: dict[str, list[dict]] = {r: [] for r in PERSON_RELATIONS}
    for r, b in zip(rows, bins):
        by_rel[r["prop"]].append({"row": r, "bin": int(b)})

    per_rel = max(1, args.n // len(PERSON_RELATIONS))
    selected: list[dict] = []
    for rel, items in by_rel.items():
        groups: dict[int, list[dict]] = {}
        for it in items:
            groups.setdefault(it["bin"], []).append(it)
        chosen: list[dict] = []
        if args.mix == "head":
            for b, its in sorted(groups.items()):
                take = min(len(its), int(round(per_rel * quint_weights[b])))
                if take:
                    idx = rng.choice(len(its), size=take, replace=False)
                    chosen.extend(its[i] for i in idx)
            # top up from whatever remains (e.g. mother relation has few items)
            if len(chosen) < per_rel:
                chosen_ids = {id(c) for c in chosen}
                pool = [it for it in items if id(it) not in chosen_ids]
                rng.shuffle(pool)
                chosen.extend(pool[: per_rel - len(chosen)])
        else:
            for b, its in sorted(groups.items()):
                take = min(len(its), args.per_bin)
                idx = rng.choice(len(its), size=take, replace=False)
                chosen.extend(its[i] for i in idx)
            rng.shuffle(chosen)
            chosen = chosen[:per_rel]
        print(f"  {rel:>16}: {len(items):>5} available -> {len(chosen):>5} sampled")
        selected.extend(chosen)

    rng.shuffle(selected)
    selected = selected[: args.n]

    questions: list[Question] = []
    for it in selected:
        r = it["row"]
        questions.append(
            Question(
                id=int(r["id"]),
                subj=str(r["subj"]),
                subj_id=int(r["subj_id"]),
                prop=str(r["prop"]),
                obj=str(r["obj"]),
                question=str(r["question"]),
                aliases=get_aliases(r),
                s_pop=int(r["s_pop"]),
                o_pop=int(r["o_pop"]),
                prompt="",  # filled in by the collector after the tokenizer is loaded
            )
        )

    with open(out_path, "w") as f:
        for q in questions:
            f.write(q.to_json() + "\n")

    print(f"Wrote {len(questions)} questions to {out_path}")
    from collections import Counter

    print("  relation counts:", dict(Counter(q.prop for q in questions)))
    print("  unique subjects:", len({q.subj_id for q in questions}))


if __name__ == "__main__":
    main()
