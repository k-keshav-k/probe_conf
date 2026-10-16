"""Run the native confidence arms A/B/C on the verified knowledge variants.

One command trains all three arms on identical examples and budgets, then
evaluates the actually emitted numbers on held-out subjects:

- draw-level Brier of the emitted report (proper score, not an external mean);
- within-question tracking across variants;
- answer accuracy with the shared computation enabled versus frozen, so a
  training-induced policy change is visible.

Examples: ``--model Qwen/Qwen2.5-0.5B-Instruct --root runs/.../seed0`` on the
laptop, or the 7B pilot on the cluster via cluster/06_native.sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.episodes import adapters_enabled, generate_answers, run_prefix
from causal_confidence.knowledge_stages import question_from_fact
from causal_confidence.native_monitor import (
    ArmConfig,
    build_grid_grammar,
    emit_report,
    train_arm,
)
from causal_confidence.outcomes import match_answer
from causal_confidence.reporters import (
    freeze_answer_policy,
    install_report_adapters,
    remove_report_adapters,
    reporter_parameters,
)
from causal_confidence.stages import load_model_and_tokenizer


def collect_items(env, facts, state_paths, layers, rank, *, draws, seed, label):
    items = []
    for variant, path in sorted(state_paths.items()):
        with _installed(env, layers, rank, path, variant):
            for index, fact in enumerate(facts):
                question = question_from_fact(fact, var.HELD_OUT_TEMPLATE)
                ids, _ = env.prefix_ids(question)
                prefix = run_prefix(env.model, ids, capture_hidden=False)
                out = generate_answers(
                    env.model, prefix, env.answer_suffix, env.tokenizer,
                    draws=draws, base_seed=seed * 1_000_003 + index * 31 + variant * 7,
                )
                matched = np.asarray(
                    [match_answer(draw.text, list(fact.references)) for draw in out],
                    dtype=np.float32,
                )
                items.append(
                    {
                        "fact_id": fact.fact_id,
                        "subject_id": fact.subject_id,
                        "variant": variant,
                        "state_path": path,
                        "input_ids": ids,
                        "suffix_ids": env.report_suffix,
                        "answer_suffix_ids": env.answer_suffix,
                        "draws": torch.tensor(matched),
                    }
                )
        print(f"  {label}: variant {variant} collected", flush=True)
    return items


class _installed:
    def __init__(self, env, layers, rank, path, variant):
        self.env, self.layers, self.rank, self.path = env, layers, rank, path
        self.adapters = None

    def __enter__(self):
        self.adapters = var.install_variant(
            self.env, layers=self.layers, rank=self.rank, state_path=self.path
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        var.remove_variant(self.env, self.adapters)
        return None


def attach_meta(items, layers, rank):
    for item in items:
        item["layers"] = layers
        item["rank"] = rank
    return items


def answer_accuracy(env, item, adapters, *, gating: bool, draws: int, seed: int) -> float:
    with _installed(env, item["layers"], item["rank"], item["state_path"], item["variant"]):
        with adapters_enabled(adapters, gating):
            prefix = run_prefix(env.model, item["input_ids"], capture_hidden=False)
        with adapters_enabled(adapters, False):
            out = generate_answers(
                env.model, prefix, item["answer_suffix_ids"], env.tokenizer,
                draws=draws, base_seed=seed,
            )
    references = item.get("references", ())
    return float(np.mean([match_answer(draw.text, list(references)) for draw in out]))


def infer_corpus_config(root: Path) -> tuple[int, int, int] | None:
    """Read subjects/train/dev counts back from the knowledge manifest.

    The variant checkpoints were trained on one corpus configuration; the
    native run must use the same one or its "test" subjects are unseen.
    """

    import re

    manifest_path = root / "manifests" / "knowledge_corpus.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())

    def subjects_of(fact_ids):
        return {int(re.search(r"_s(\d+)_", fact_id).group(1)) for fact_id in fact_ids}

    train = subjects_of(manifest["splits"]["train"])
    dev = subjects_of(manifest["splits"]["dev"])
    test = subjects_of(manifest["splits"]["test"])
    subjects = max(train | dev | test) + 1
    return subjects, len(train), len(dev)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--root", required=True, help="knowledge run dir with variants/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-answer", type=float, default=1.0)
    parser.add_argument("--train-draws", type=int, default=8)
    parser.add_argument("--eval-draws", type=int, default=16)
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--train-subjects", type=int, default=None)
    parser.add_argument("--dev-subjects", type=int, default=None)
    parser.add_argument("--accuracy-items", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--reporter-layers",
        default=None,
        help="comma-separated decoder layers for the reporter adapter; "
        "default is the middle layer",
    )
    parser.add_argument("--reporter-rank", type=int, default=8)
    args = parser.parse_args()
    if args.subjects is None or args.train_subjects is None or args.dev_subjects is None:
        inferred = infer_corpus_config(Path(args.root))
        if inferred is not None:
            args.subjects = args.subjects or inferred[0]
            args.train_subjects = args.train_subjects or inferred[1]
            args.dev_subjects = args.dev_subjects or inferred[2]
            print(f"inferred corpus config from manifest: {inferred}", flush=True)
        else:
            args.subjects = args.subjects or 60
            args.train_subjects = args.train_subjects or 36
            args.dev_subjects = args.dev_subjects or 12

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = load_model_and_tokenizer(args.model, revision=None, device=args.device, dtype="auto")
    n_layers = int(env.model.config.num_hidden_layers)
    variant_layers = var.default_adapter_layers(n_layers)
    if args.reporter_layers:
        reporter_layers = tuple(int(item) for item in args.reporter_layers.split(","))
    else:
        reporter_layers = (n_layers // 2,)

    facts = know.build_corpus(args.subjects, seed=args.seed)
    splits = know.split_by_subject(
        facts, train_subjects=args.train_subjects, dev_subjects=args.dev_subjects
    )
    root = Path(args.root)
    state_paths = {v: str(root / "variants" / f"variant_{v}.pt") for v in know.VARIANT_IDS}
    grammar = build_grid_grammar(env.tokenizer)

    print("collecting training items", flush=True)
    train_items = attach_meta(
        collect_items(env, splits["train"], state_paths, variant_layers, 8,
                      draws=args.train_draws, seed=args.seed, label="train"),
        variant_layers, 8,
    )
    print("collecting test items", flush=True)
    test_items = attach_meta(
        collect_items(env, splits["test"], state_paths, variant_layers, 8,
                      draws=args.eval_draws, seed=args.seed + 2, label="test"),
        variant_layers, 8,
    )
    for item in train_items + test_items:
        item["question"] = question_from_fact(
            next(f for f in facts if f.fact_id == item["fact_id"]), var.HELD_OUT_TEMPLATE
        )
        item["references"] = next(f for f in facts if f.fact_id == item["fact_id"]).references

    pooled: dict[str, torch.Tensor] = {}
    for item in train_items:
        pooled.setdefault(item["fact_id"], []).append(item["draws"])
    pooled = {fact_id: torch.cat(values) for fact_id, values in pooled.items()}

    results: dict = {
        "model": args.model,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "beta_answer": args.beta_answer,
        "reporter_layers": list(reporter_layers),
        "reporter_rank": args.reporter_rank,
        "n_train_items": len(train_items),
        "n_test_items": len(test_items),
        "arms": {},
    }
    for arm in args.arms:
        print(f"=== training arm {arm} ===", flush=True)
        freeze_answer_policy(env.model)
        reporter = install_report_adapters(env.model, reporter_layers, rank=args.reporter_rank)
        for parameter in reporter_parameters(env.model):
            parameter.requires_grad_(True)
        arm_items = [
            {**item, "draws": pooled[item["fact_id"]] if arm == "C" else item["draws"]}
            for item in train_items
        ]
        training = train_arm(
            env,
            reporter_adapters=reporter,
            prefix_and_suffix=arm_items,
            grammar=grammar,
            config=ArmConfig(
                arm=arm, steps=args.steps, warmup_steps=args.warmup_steps, lr=args.lr,
                beta_answer=args.beta_answer, seed=args.seed, log_every=max(1, args.steps // 5),
            ),
        )
        # Emitted reports on held-out subjects.
        emitted = []
        for item in test_items:
            report = emit_report(env, item, grammar, arm=arm, reporter_adapters=reporter)
            emitted.append(
                {
                    "fact_id": item["fact_id"],
                    "subject_id": item["subject_id"],
                    "variant": item["variant"],
                    "value": report["value"],
                    "valid": report["valid"],
                    "draws": item["draws"].tolist(),
                }
            )
        values = np.asarray([row["value"] for row in emitted], dtype=np.float64)
        draws_matrix = np.stack([np.asarray(row["draws"], dtype=np.float64) for row in emitted])
        brier = float(((values[:, None] - draws_matrix) ** 2).mean())
        emitted_success = draws_matrix.mean(axis=1)
        # Within-question tracking across variants.
        spread, contrast = [], []
        by_fact: dict[str, list[int]] = {}
        for index, row in enumerate(emitted):
            by_fact.setdefault(row["fact_id"], []).append(index)
        for indices in by_fact.values():
            if len(indices) < 2:
                continue
            v = values[indices]
            s = emitted_success[indices]
            spread.append(float(np.corrcoef(v, s)[0, 1]) if v.std() > 0 and s.std() > 0 else 0.0)
            difference = (v[:, None] - v[None, :]) - (s[:, None] - s[None, :])
            contrast.append(float((difference**2).mean()))
        # Answer accuracy with the shared computation on versus frozen.
        stride = max(1, len(test_items) // max(1, args.accuracy_items))
        accuracy_subset = test_items[::stride][: args.accuracy_items]
        accuracy_enabled, accuracy_frozen = [], []
        for item in accuracy_subset:
            accuracy_enabled.append(
                answer_accuracy(env, item, reporter, gating=arm in ("B", "C"), draws=4, seed=args.seed + 5)
            )
            accuracy_frozen.append(
                answer_accuracy(env, item, reporter, gating=False, draws=4, seed=args.seed + 5)
            )
        results["arms"][arm] = {
            "training": training,
            "emitted_brier": brier,
            "emitted_valid_rate": float(np.mean([row["valid"] for row in emitted])),
            "within_question_correlation": float(np.mean(spread)) if spread else None,
            "within_question_contrast": float(np.mean(contrast)) if contrast else None,
            "answer_accuracy_enabled": float(np.mean(accuracy_enabled)),
            "answer_accuracy_frozen": float(np.mean(accuracy_frozen)),
            "emitted": emitted,
        }
        print(
            f"arm {arm}: emitted Brier {brier:.4f} valid {results['arms'][arm]['emitted_valid_rate']:.3f} "
            f"corr {results['arms'][arm]['within_question_correlation']} "
            f"acc enabled {results['arms'][arm]['answer_accuracy_enabled']:.3f} "
            f"frozen {results['arms'][arm]['answer_accuracy_frozen']:.3f}",
            flush=True,
        )
        remove_report_adapters(env.model, reporter)
    out_path.write_text(json.dumps(results, indent=2))
    print("written", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
