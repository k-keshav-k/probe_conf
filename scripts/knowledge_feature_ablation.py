"""Feature ablation for the knowledge-variant reporter (diagnostic).

Question: is the negative added value a property of the task (probability
features already capture everything) or an artifact of the reporter head
drowning in the large hidden block?  Trains the same T1/T2 heads on
question-only, question+prefix, question+hidden, and all blocks, and compares
each with the strongest ordinary baseline fitted on the same states.

Run on seed 0's cached knowledge variants; writes a JSON summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.baselines import BaselineFeatures
from causal_confidence.knowledge_stages import (
    StateData,
    baseline_and_added_value,
    collect_all_variants,
    compare_reporters,
    evaluate_reporter,
    select_lambda,
)
from causal_confidence.stages import load_model_and_tokenizer


def subset(data: StateData, blocks: tuple[str, ...]) -> StateData:
    return StateData(
        fact_ids=data.fact_ids,
        subject_ids=data.subject_ids,
        variant_ids=data.variant_ids,
        features=BaselineFeatures(
            {key: data.features.blocks[key] for key in blocks if key in data.features.blocks}
        ),
        outcomes=data.outcomes,
    )


def main() -> int:
    seed = 0
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/knowledge_ablation.json")
    base = Path(f"runs/causal_confidence/cc_v1/knowledge/seed{seed}")
    env = load_model_and_tokenizer(
        "Qwen/Qwen2.5-0.5B-Instruct", revision=None, device="cuda", dtype="auto"
    )
    facts = know.build_corpus(60, seed=seed)
    exposure = know.assign_exposure(facts)
    splits = know.split_by_subject(facts, train_subjects=36, dev_subjects=12)
    layers = var.default_adapter_layers(24)
    state_paths = {v: str(base / "variants" / f"variant_{v}.pt") for v in know.VARIANT_IDS}

    print("collecting train states", flush=True)
    train = collect_all_variants(
        env, layers=layers, rank=8,
        variant_states={v: state_paths[v] for v in know.VARIANT_IDS},
        facts=splits["train"], template=var.HELD_OUT_TEMPLATE,
        draws=8, seed=seed, partition="ablation_train",
    )
    print("collecting dev/test states", flush=True)
    dev = collect_all_variants(
        env, layers=layers, rank=8,
        variant_states={v: state_paths[v] for v in know.VARIANT_IDS},
        facts=splits["dev"], template=var.HELD_OUT_TEMPLATE,
        draws=8, seed=seed + 1, partition="ablation_dev",
    )
    test = collect_all_variants(
        env, layers=layers, rank=8,
        variant_states={v: state_paths[v] for v in know.VARIANT_IDS},
        facts=splits["test"], template=var.HELD_OUT_TEMPLATE,
        draws=16, seed=seed + 2, partition="ablation_test",
    )

    results = {}
    for blocks in (
        ("question",),
        ("question", "prefix"),
        ("question", "hidden"),
        ("question", "prefix", "hidden"),
    ):
        label = "+".join(blocks)
        tr, dv, te = subset(train, blocks), subset(dev, blocks), subset(test, blocks)
        selection = select_lambda(tr, dv, lambdas=(0.25, 1.0, 4.0), steps=1_200, seed=seed)
        t1 = selection["heads"]["0"]
        t2 = selection["heads"][str(selection["best_lambda"])]
        report_t1 = evaluate_reporter(t1, te, label="t1")
        report_t2 = evaluate_reporter(t2, te, label="t2")
        comparison = compare_reporters(t1, t2, te, seed=seed)
        added = baseline_and_added_value(
            tr, dv, te,
            t2.expected(tr.reporter_matrix()),
            t2.expected(te.reporter_matrix()),
            seed=seed,
        )
        results[label] = {
            "best_lambda": selection["best_lambda"],
            "t1_brier": report_t1["brier"],
            "t2_brier": report_t2["brier"],
            "delta_brier": comparison["delta_brier"],
            "t1_contrast": report_t1["contrast_error"],
            "t2_contrast": report_t2["contrast_error"],
            "t2_slope": report_t2["calibration"]["slope"],
            "added_value": added["added_value"],
            "strongest_baseline": added["strongest_baseline"],
        }
        print(
            f"{label:28s} lambda={selection['best_lambda']} "
            f"t1={report_t1['brier']:.4f} t2={report_t2['brier']:.4f} "
            f"baseline={added['added_value']['baseline_brier']:.4f} "
            f"added={added['added_value']['added']:+.4f} "
            f"slope={report_t2['calibration']['slope']:+.3f}",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("written", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
