"""Smoke test: GPU stack, chat template, hooks, and a 3-question mini-run.

Usage:
    .venv/bin/python scripts/smoke_test.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import Question, SYSTEM_PROMPT  # noqa: E402
from scripts.collect import (  # noqa: E402
    build_prompts,
    extract_features,
    generate_confidence,
    generate_correctness,
    get_device,
    get_layers,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    print("=" * 60)
    print("1. GPU check")
    device = get_device()
    print(f"   device={device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"   {torch.cuda.get_device_name(0)} | {p.total_memory/1e9:.1f} GB | arch {torch.cuda.get_arch_list()}")

    print("=" * 60)
    print("2. Load tokenizer/model")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    print(f"   loaded in {time.time()-t0:.1f}s; layers={model.config.num_hidden_layers}, "
          f"d_ff={model.config.intermediate_size}")

    print("=" * 60)
    print("3. Chat template consistency")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Who is the father of Kim Kardashian?"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tmpl_out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # transformers 5.x may return a BatchEncoding here
    ids_tmpl = tmpl_out["input_ids"] if hasattr(tmpl_out, "keys") else tmpl_out
    ids_manual = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"   template ids == manual ids: {ids_tmpl == ids_manual} ({len(ids_manual)} tokens)")
    assert ids_tmpl == ids_manual, "chat template tokenization mismatch"

    print("=" * 60)
    print("4. Mini end-to-end (3 questions, k=%d)" % args.k)
    qs = [
        Question(1, "Kim Kardashian", 1, "father", "Robert Kardashian",
                 "Who is the father of Kim Kardashian?", ["Robert Kardashian"], 1000, 10),
        Question(2, "Albert Einstein", 2, "place of birth", "Ulm",
                 "In what city was Albert Einstein born?", ["Ulm"], 5000, 100),
        Question(3, "Winston Churchill", 3, "mother", "Jennie Jerome",
                 "Who is the mother of Winston Churchill?", ["Jennie Jerome"], 3000, 50),
    ]
    build_prompts(tokenizer, qs)
    frac, greedy, samples = generate_correctness(model, tokenizer, qs, args.k, 2, device)
    conf = generate_confidence(model, tokenizer, qs, 4, device)
    print(f"   frac_correct={frac.tolist()}")
    print(f"   greedy={greedy.tolist()}")
    print(f"   confidence={conf.tolist()}")
    for q, ss in zip(qs, samples):
        print(f"   [{q.prop}] {q.question} -> {[s.strip()[:40] for s in ss]}")

    print("=" * 60)
    print("5. Feature extraction (hooks)")
    t0 = time.time()
    rows = extract_features(model, tokenizer, qs, 2, device, "smoke", save_raw=True)
    print(f"   extracted in {time.time()-t0:.1f}s; rows={len(rows)}, cols={len(rows[0])}")
    for key in sorted(rows[0])[:12]:
        print(f"     {key}: {rows[0][key] if not isinstance(rows[0][key], float) else round(rows[0][key], 4)}")
    raw = np.load(sorted(Path("results/raw").glob("smoke_batch*.npz"))[-1])
    print("   raw shapes:", {k: v.shape for k, v in raw.items()})
    print("=" * 60)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
