"""Integration check for the retained-state fork on a real model.

Runs one prefix, forks the cache into an answer branch and a confidence branch,
and compares each branch with an independent full run.  Then applies a
mid-layer intervention on a subject token and confirms that both branches see
the change.  The script exits non-zero on failure.

Usage:

    python scripts/check_retained_state.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from section33.analysis.retained_state import (  # noqa: E402
    additive_transform,
    continue_branch,
    run_prefix,
)
from section33.models.loader import chat_prompt_ids, load_model_and_tokenizer  # noqa: E402

QUESTION = "In what city was Marie Curie born?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--token", type=int, default=-2)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    loaded = load_model_and_tokenizer(args.model, dtype=args.dtype)
    model = loaded.model
    tokenizer = loaded.tokenizer
    layer = args.layer
    if layer is None:
        layer = max(0, model.config.num_hidden_layers // 2 - 2)

    prefix_ids = chat_prompt_ids(tokenizer, QUESTION)
    prefix = torch.tensor([prefix_ids], device=loaded.device)
    answer_suffix = torch.tensor(
        [tokenizer.encode(" Paris", add_special_tokens=False)], device=loaded.device
    )
    confidence_suffix = torch.tensor(
        [tokenizer.encode(" 85", add_special_tokens=False)], device=loaded.device
    )

    def full_run(suffix: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([prefix, suffix], dim=1)
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits[:, -suffix.shape[1] :]

    state = run_prefix(model, prefix)
    answer = continue_branch(model, state, answer_suffix)
    confidence = continue_branch(model, state, confidence_suffix)
    answer_full = full_run(answer_suffix)
    confidence_full = full_run(confidence_suffix)
    tolerance = 1e-3 if args.dtype == "float32" else 0.1

    def compare(branch: torch.Tensor, reference: torch.Tensor) -> dict:
        return {
            "max_diff": float((branch - reference).abs().max()),
            "top1_match": float(
                (branch.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean()
            ),
        }

    answer_check = compare(answer.logits, answer_full)
    confidence_check = compare(confidence.logits, confidence_full)
    checks = {
        "answer_full": answer_check,
        "confidence_full": confidence_check,
        "tolerance": tolerance,
        "prefix_len": state.prefix_len,
        "num_layers": model.config.num_hidden_layers,
        "layer": layer,
        "token": args.token,
        "dtype": args.dtype,
    }

    vector = torch.zeros(model.config.hidden_size, device=loaded.device, dtype=torch.float32)
    vector[0] = 5.0
    intervened = run_prefix(
        model,
        prefix,
        layer_index=layer,
        transform=additive_transform(vector.to(next(model.parameters()).dtype), token_index=args.token),
    )
    answer_int = continue_branch(model, intervened, answer_suffix)
    confidence_int = continue_branch(model, intervened, confidence_suffix)
    checks["answer_intervention_delta"] = float(
        (answer_int.logits - answer.logits).abs().max()
    )
    checks["confidence_intervention_delta"] = float(
        (confidence_int.logits - confidence.logits).abs().max()
    )

    last_layer = run_prefix(
        model,
        prefix,
        layer_index=model.config.num_hidden_layers - 1,
        transform=additive_transform(vector.to(next(model.parameters()).dtype), token_index=args.token),
    )
    answer_last = continue_branch(model, last_layer, answer_suffix)
    checks["last_layer_delta"] = float(
        (answer_last.logits - answer.logits).abs().max()
    )

    checks["pass"] = bool(
        answer_check["max_diff"] < tolerance
        and confidence_check["max_diff"] < tolerance
        and answer_check["top1_match"] == 1.0
        and confidence_check["top1_match"] == 1.0
        and checks["answer_intervention_delta"] > 1e-3
        and checks["confidence_intervention_delta"] > 1e-3
    )
    print(json.dumps(checks, indent=2))
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
