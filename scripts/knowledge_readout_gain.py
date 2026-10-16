"""What produces the readout gain? (T1 vs T2, probability-only vs +hidden.)

Runs the corrected evaluation (draw-level Brier, out-of-fold added value)
for four reporter configurations on seed-0 states:

| inputs | objective |
| --- | --- |
| question + prefix probabilities | T1 |
| question + prefix probabilities | T2 (lambda on dev) |
| question + prefix + raw hidden | T1 |
| question + prefix + raw hidden | T2 (lambda on dev) |

Positive added value with `question+prefix` would mean the gain is a more
flexible transform of the baseline features; a gain only with hidden would
mean additional internal-state information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.baselines import BaselineFeatures
from causal_confidence.knowledge_stages import (
    StateData,
    baseline_and_added_value,
    evaluate_reporter,
    fit_pair_reporter,
    out_of_fold_report_predictions,
    reporter_predictions,
    select_lambda,
)
from causal_confidence.stages import load_model_and_tokenizer
from knowledge_readout_audit import collect


def subset(data: StateData, blocks: tuple[str, ...]) -> StateData:
    return StateData(
        fact_ids=data.fact_ids,
        subject_ids=data.subject_ids,
        variant_ids=data.variant_ids,
        features=BaselineFeatures(
            {key: value for key, value in data.features.blocks.items() if key in blocks}
        ),
        outcomes=data.outcomes,
    )


def main() -> int:
    seed = 0
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/knowledge_readout_gain.json")
    base = Path(f"runs/causal_confidence/cc_v1/knowledge/seed{seed}")
    env = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", revision=None, device="cuda", dtype="auto")
    facts = know.build_corpus(60, seed=seed)
    splits = know.split_by_subject(facts, train_subjects=36, dev_subjects=12)
    layers = var.default_adapter_layers(24)
    state_paths = {v: str(base / "variants" / f"variant_{v}.pt") for v in know.VARIANT_IDS}

    print("collecting states", flush=True)
    train = collect(env, splits["train"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 8, seed, "train")
    dev = collect(env, splits["dev"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 8, seed + 1, "dev")
    test = collect(env, splits["test"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 16, seed + 2, "test")

    def to_state(data: dict) -> StateData:
        return StateData(
            fact_ids=list(data["fact_ids"]),
            subject_ids=list(data["subject_ids"]),
            variant_ids=list(data["variant_ids"]),
            features=BaselineFeatures(
                {"question": data["question"], "prefix": data["prefix"], "hidden": data["hidden"]}
            ),
            outcomes=data["outcomes"],
        )

    train_states, dev_states, test_states = to_state(train), to_state(dev), to_state(test)
    results: dict = {}
    for blocks in (("question", "prefix"), ("question", "prefix", "hidden")):
        label = "+".join(blocks)
        tr, dv, te = subset(train_states, blocks), subset(dev_states, blocks), subset(test_states, blocks)
        for objective in ("T1", "T2"):
            if objective == "T1":
                lam, head = 0.0, fit_pair_reporter(tr, lam=0.0, steps=1_200, seed=seed)
            else:
                selection = select_lambda(tr, dv, lambdas=(0.25, 1.0, 4.0), steps=1_200, seed=seed)
                lam = float(selection["best_lambda"])
                head = selection["heads"][str(lam)]
            oof = out_of_fold_report_predictions(
                tr, lam=lam, steps=1_200, seed=seed
            )
            added = baseline_and_added_value(
                tr, dv, te, oof, reporter_predictions(head, te), seed=seed
            )
            report = evaluate_reporter(head, te, label=f"{label}_{objective}")
            key = f"{label}|{objective}"
            results[key] = {
                "lambda": lam,
                "brier": report["brier"],
                "auroc": report["auroc"],
                "added": added["added_value"]["added"],
                "added_ci": added["added_value"]["ci"],
                "baseline_brier": added["added_value"]["baseline_brier"],
                "strongest_baseline": added["strongest_baseline"],
            }
            print(
                f"{key:34s} lambda {lam:<4} brier {report['brier']:.4f} auroc {report['auroc']:.3f} "
                f"added {added['added_value']['added']:+.4f} "
                f"CI [{added['added_value']['ci']['ci_low']:+.4f},{added['added_value']['ci']['ci_high']:+.4f}] "
                f"baseline {added['added_value']['baseline_brier']:.4f}",
                flush=True,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("written", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
