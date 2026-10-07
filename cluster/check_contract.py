"""GPU contract check for one model before collection.

Verifies, on a real GPU node:
  - the MLP projections have no bias,
  - ``W_down s`` reconstructs the module output,
  - the frozen feature families extract reference-free features,
  - the feature count is in the expected range.

Usage:
    python cluster/check_contract.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json

import torch

from section33.models.collect import ModelWeights, extract_features_for_sequences
from section33.models.contract import verify_mlp_contract
from section33.models.loader import load_model_and_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    loaded = load_model_and_tokenizer(
        args.model, revision=args.revision, dtype="float16", device=args.device
    )
    report: dict = {
        "model": loaded.name,
        "revision": loaded.revision,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "layers": loaded.model.config.num_hidden_layers,
    }
    report["contract"] = verify_mlp_contract(loaded.model)

    weights = ModelWeights(loaded.model, loaded.device)
    prompt = "Who is the father of George W. Bush?"
    ids = loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    features = extract_features_for_sequences(
        loaded, weights, [ids], [[len(ids) - 1]], ["last_prompt"], probability=True
    )[0]
    report["n_features"] = len(features)
    report["families"] = sorted({key.split("__")[0] for key in features})
    report["reference_free"] = not any(
        name in key for key in features
        for name in ("cos_sel_agold", "proj_gold_norm", "mlp_margin", "ent_cand", "logit_mlp_gold")
    )
    report["passed"] = bool(
        report["contract"]["passed"] and report["reference_free"] and report["n_features"] > 1000
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
