"""Readout audit for the knowledge-variant study (experiment 1).

The user-facing question: is the negative hidden-state result a fact about
information, or a fact about learning to read it?  Three checks:

1. **Reconstruction gate.** The answer-fork logits are a function of the final
   hidden state through the model's frozen output head.  Rebuild the logits
   from the saved hidden state and compare (pre-norm vs post-norm head).
2. **Probe reproduction.** Fit a probe on the raw hidden state to reproduce
   the probability baseline's predictions with subject-grouped cross-fitting.
   Gate: held-out reproduction MSE at most 0.001.  If the probe reproduces the
   baseline, the hidden state carries the baseline's information.
3. **Corrected reporter value.** Draw-level Brier and out-of-fold report
   predictions for the added-value comparison.

Runs on seed 0's cached knowledge variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.baselines import (
    BaselineFeatures,
    fit_predict_outcomes,
    prefix_probability_features,
    question_features,
    rich_prefix_features,
)
from causal_confidence.episodes import generate_answers, run_prefix, suffix_state
from causal_confidence.knowledge_stages import (
    StateData,
    baseline_and_added_value,
    compare_reporters,
    evaluate_reporter,
    installed_variant,
    out_of_fold_report_predictions,
    question_from_fact,
    reporter_predictions,
    select_lambda,
)
from causal_confidence.outcomes import match_answer
from causal_confidence.stages import load_model_and_tokenizer


def collect(env, facts, state_paths, layers, rank, template, draws, seed, label) -> dict:
    """States with raw hidden, true fork logits (transient), and draw labels."""

    data = {
        "question": [], "prefix": [], "rich": [], "hidden": [],
        "outcomes": [], "fact_ids": [], "subject_ids": [], "variant_ids": [],
    }
    reconstruction = {"max_abs_diff_norm_then_head": 0.0, "max_abs_diff_head_only": 0.0}
    final_norm = env.model.model.norm
    lm_head = env.model.lm_head
    for variant, path in sorted(state_paths.items()):
        with installed_variant(env, layers=layers, rank=rank, state_path=path, variant_id=variant):
            for index, fact in enumerate(facts):
                question = question_from_fact(fact, template)
                ids, digest = env.prefix_ids(question)
                prefix = run_prefix(env.model, ids, capture_hidden=False, prompt_hash_value=digest)
                logits, hidden = suffix_state(env.model, prefix, env.answer_suffix)
                with torch.no_grad():
                    # Weights are fp16, so compute in the model dtype and
                    # compare in float32.
                    reclaimed = lm_head(final_norm(hidden)).float()
                    head_only = lm_head(hidden).float()
                    reconstruction["max_abs_diff_norm_then_head"] = max(
                        reconstruction["max_abs_diff_norm_then_head"],
                        float((reclaimed - logits.float()).abs().max()),
                    )
                    reconstruction["max_abs_diff_head_only"] = max(
                        reconstruction["max_abs_diff_head_only"],
                        float((head_only - logits.float()).abs().max()),
                    )
                draws_out = generate_answers(
                    env.model, prefix, env.answer_suffix, env.tokenizer,
                    draws=draws, base_seed=seed * 1_000_003 + index * 31 + variant * 7,
                )
                matched = [match_answer(draw.text, list(fact.references)) for draw in draws_out]
                vector = logits[0].float().cpu().numpy()
                data["question"].append(question_features([fact.attribute], [None], [question.question_text]))
                data["prefix"].append(prefix_probability_features(vector[None, :]))
                data["rich"].append(rich_prefix_features(vector[None, :]))
                data["hidden"].append(hidden[0].float().cpu().numpy())
                data["outcomes"].append(np.asarray(matched, dtype=np.float32))
                data["fact_ids"].append(fact.fact_id)
                data["subject_ids"].append(fact.subject_id)
                data["variant_ids"].append(variant)
        print(f"  {label}: variant {variant}", flush=True)
    return {
        "question": np.concatenate(data["question"], axis=0),
        "prefix": np.concatenate(data["prefix"], axis=0),
        "rich": np.concatenate(data["rich"], axis=0),
        "hidden": np.stack(data["hidden"]),
        "outcomes": np.stack(data["outcomes"]),
        "fact_ids": np.asarray(data["fact_ids"]),
        "subject_ids": np.asarray(data["subject_ids"]),
        "variant_ids": np.asarray(data["variant_ids"]),
        "reconstruction": reconstruction,
    }


def cross_fitted_probe(
    train_hidden, train_target, test_hidden, test_target, *, kind="ridge", folds=4, seed=0
):
    """Subject-fold cross-fitting is handled by the caller; here hidden->target."""

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(128,), max_iter=400,
                                 early_stopping=True, random_state=seed)),
        ])
    model.fit(train_hidden, train_target)
    return np.clip(model.predict(test_hidden), 0.0, 1.0)


def subject_folds(subjects, folds, seed):
    unique = list(dict.fromkeys(subjects))
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    mapping = {subject: index % folds for index, subject in enumerate(unique)}
    return np.asarray([mapping[subject] for subject in subjects])


def main() -> int:
    seed = 0
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/knowledge_readout_audit.json")
    base = Path(f"runs/causal_confidence/cc_v1/knowledge/seed{seed}")
    env = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", revision=None, device="cuda", dtype="auto")
    facts = know.build_corpus(60, seed=seed)
    splits = know.split_by_subject(facts, train_subjects=36, dev_subjects=12)
    layers = var.default_adapter_layers(24)
    state_paths = {v: str(base / "variants" / f"variant_{v}.pt") for v in know.VARIANT_IDS}

    print("collecting train", flush=True)
    train = collect(env, splits["train"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 8, seed, "train")
    print("collecting dev", flush=True)
    dev = collect(env, splits["dev"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 8, seed + 1, "dev")
    print("collecting test", flush=True)
    test = collect(env, splits["test"], state_paths, layers, 8, var.HELD_OUT_TEMPLATE, 16, seed + 2, "test")

    results: dict = {
        "reconstruction": {
            "train": train["reconstruction"],
            "dev": dev["reconstruction"],
            "test": test["reconstruction"],
        }
    }
    print("reconstruction:", json.dumps(results["reconstruction"]), flush=True)

    # --- probability baseline ---------------------------------------------------
    train_features = BaselineFeatures({"question": train["question"], "prefix": train["prefix"]})
    test_features = BaselineFeatures({"question": test["question"], "prefix": test["prefix"]})
    baseline_train = fit_predict_outcomes(train_features.stack(["question", "prefix"]),
                                          train["outcomes"],
                                          train_features.stack(["question", "prefix"]),
                                          kind="logistic", seed=seed)
    baseline_test = fit_predict_outcomes(train_features.stack(["question", "prefix"]),
                                         train["outcomes"],
                                         test_features.stack(["question", "prefix"]),
                                         kind="logistic", seed=seed)
    baseline_brier = float(((baseline_test[:, None] - test["outcomes"]) ** 2).mean())
    results["baseline"] = {"test_brier": baseline_brier}
    print("baseline test Brier:", round(baseline_brier, 4), flush=True)

    # --- probe reproduction of the baseline predictions, cross-fitted ----------
    fold_of = subject_folds(test["subject_ids"], folds=4, seed=seed)
    for kind in ("ridge", "mlp"):
        reproduced = np.full(len(test["subject_ids"]), np.nan)
        direct = np.full(len(test["subject_ids"]), np.nan)
        for fold in range(4):
            train_mask = subject_folds(train["subject_ids"], folds=4, seed=seed) != fold
            test_mask = fold_of == fold
            if not test_mask.any():
                continue
            reproduced[test_mask] = cross_fitted_probe(
                train["hidden"][train_mask], baseline_train[train_mask],
                test["hidden"][test_mask], None, kind=kind, seed=seed,
            )
            direct[test_mask] = cross_fitted_probe(
                train["hidden"][train_mask], train["outcomes"][train_mask].mean(axis=1),
                test["hidden"][test_mask], None, kind=kind, seed=seed,
            )
        reproduction_mse = float(((reproduced - baseline_test) ** 2).mean())
        reproduced_brier = float(((reproduced[:, None] - test["outcomes"]) ** 2).mean())
        direct_brier = float(((direct[:, None] - test["outcomes"]) ** 2).mean())
        results[f"probe_{kind}"] = {
            "reproduction_mse": reproduction_mse,
            "reproduction_brier": reproduced_brier,
            "direct_brier": direct_brier,
        }
        print(
            f"probe {kind}: reproduce-MSE {reproduction_mse:.4f} "
            f"| Brier(reproduced baseline) {reproduced_brier:.4f} "
            f"| Brier(direct success) {direct_brier:.4f}",
            flush=True,
        )

    # --- corrected reporter value ----------------------------------------------
    def to_state(data: dict) -> StateData:
        return StateData(
            fact_ids=list(data["fact_ids"]),
            subject_ids=list(data["subject_ids"]),
            variant_ids=list(data["variant_ids"]),
            features=BaselineFeatures({
                "question": data["question"], "prefix": data["prefix"], "hidden": data["hidden"]
            }),
            outcomes=data["outcomes"],
        )

    train_states, dev_states, test_states = to_state(train), to_state(dev), to_state(test)
    selection = select_lambda(train_states, dev_states, lambdas=(0.25, 1.0, 4.0), steps=1_200, seed=seed)
    t1 = selection["heads"]["0"]
    t2 = selection["heads"][str(selection["best_lambda"])]
    comparison = compare_reporters(t1, t2, test_states, seed=seed)
    oof = out_of_fold_report_predictions(
        train_states, lam=float(selection["best_lambda"]), steps=1_200, seed=seed
    )
    added = baseline_and_added_value(
        train_states, dev_states, test_states, oof,
        reporter_predictions(t2, test_states), seed=seed,
    )
    results["reporter"] = {
        "best_lambda": selection["best_lambda"],
        "t1": {k: evaluate_reporter(t1, test_states, label="t1")[k]
               for k in ("brier", "brier_vs_success_mean", "calibration")},
        "t2": {k: evaluate_reporter(t2, test_states, label="t2")[k]
               for k in ("brier", "brier_vs_success_mean", "calibration")},
        "comparison": comparison,
        "added_value": added["added_value"],
    }
    print("reporter T2 added (OOF, draw-level):", round(added["added_value"]["added"], 4),
          "CI", added["added_value"]["ci"], flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("written", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
