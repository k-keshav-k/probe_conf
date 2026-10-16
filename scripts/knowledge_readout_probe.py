"""Hidden-state and probability-feature readout probe (diagnostic).

The feature ablation showed the report head is best without the seeded hidden
projection, and that a `question+prefix` head does not beat the prefix
baseline.  Two fairness questions remain:

1. Does the richer prefix block (top-k logits + hashed identities) beat the
   summary prefix block?  (The added-value comparison excluded `rich`.)
2. Is the hidden projection lossy, or is the hidden state genuinely
   uninformative beyond the prefix?  Here we use the raw final hidden state
   (896 dims at 0.5B) and fit a ridge readout.

State collection is the same episode path as the study; only the readouts
change.  Runs on seed 0's cached knowledge variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.baselines import (
    prefix_probability_features,
    question_features,
    rich_prefix_features,
)
from causal_confidence.episodes import generate_answers, run_prefix, suffix_state
from causal_confidence.knowledge_stages import installed_variant, question_from_fact
from causal_confidence.outcomes import match_answer
from causal_confidence.stages import load_model_and_tokenizer


def collect(env, facts, state_paths, layers, rank, template, draws, seed, label):
    data = {
        "question": [], "prefix": [], "rich": [], "hidden": [],
        "outcomes": [], "fact_ids": [], "variant_ids": [],
    }
    for variant, path in sorted(state_paths.items()):
        with installed_variant(env, layers=layers, rank=rank, state_path=path, variant_id=variant):
            for index, fact in enumerate(facts):
                question = question_from_fact(fact, template)
                ids, digest = env.prefix_ids(question)
                prefix = run_prefix(env.model, ids, capture_hidden=False, prompt_hash_value=digest)
                logits, hidden = suffix_state(env.model, prefix, env.answer_suffix)
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
                data["outcomes"].append(float(np.mean(matched)))
                data["fact_ids"].append(fact.fact_id)
                data["variant_ids"].append(variant)
        print(f"  {label}: variant {variant} collected", flush=True)
    return {
        "question": np.concatenate(data["question"], axis=0),
        "prefix": np.concatenate(data["prefix"], axis=0),
        "rich": np.concatenate(data["rich"], axis=0),
        "hidden": np.stack(data["hidden"]),
        "outcomes": np.asarray(data["outcomes"]),
        "fact_ids": np.asarray(data["fact_ids"]),
        "variant_ids": np.asarray(data["variant_ids"]),
    }


def ridge_fit_predict(x_train, y_train, x_test, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_test), 0.0, 1.0)


def mlp_fit_predict(x_train, y_train, x_dev, y_dev, x_test, *, seed=0, steps=1500, hidden=128):
    """Small MLP readout with dev-selected weight decay (bounded nonlinear check)."""

    import torch

    torch.set_num_threads(4)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    xd = torch.tensor((x_dev - mean) / std, dtype=torch.float32)
    xs = torch.tensor((x_test - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    yd = torch.tensor(y_dev, dtype=torch.float32)
    best = None
    for weight_decay in (1e-5, 1e-4, 1e-3):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(xt.shape[1], hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
        for _ in range(steps):
            index = torch.randint(0, len(xt), (256,))
            prediction = net(xt[index]).squeeze(1)
            loss = (prediction - yt[index]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            dev_mse = float((net(xd).squeeze(1) - yd).pow(2).mean())
        if best is None or dev_mse < best[0]:
            best = (dev_mse, weight_decay, net)
    _, weight_decay, net = best
    with torch.no_grad():
        prediction = net(xs).squeeze(1).clamp(0.0, 1.0).numpy()
    return prediction, {"dev_mse": best[0], "weight_decay": weight_decay}


def main() -> int:
    import os

    seed = 0
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/knowledge_readout_probe.json")
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

    y_train = train["outcomes"]
    y_dev = dev["outcomes"]
    y_test = test["outcomes"]
    sets = {
        "question": ["question"],
        "prefix": ["prefix"],
        "rich": ["rich"],
        "hidden_raw": ["hidden"],
        "question+prefix": ["question", "prefix"],
        "prefix+rich": ["prefix", "rich"],
        "prefix+hidden_raw": ["prefix", "hidden"],
        "prefix+rich+hidden_raw": ["prefix", "rich", "hidden"],
    }
    results = {}
    for name, blocks in sets.items():
        x_train = np.concatenate([train[block] for block in blocks], axis=1).astype(np.float32)
        x_test = np.concatenate([test[block] for block in blocks], axis=1).astype(np.float32)
        prediction = ridge_fit_predict(x_train, y_train, x_test)
        mse = float(((prediction - y_test) ** 2).mean())
        results[name] = {"test_squared_error": mse, "n_features": int(x_train.shape[1])}
        print(f"ridge {name:28s} features={x_train.shape[1]:4d} test_mse={mse:.4f}", flush=True)
    # Bounded nonlinear check on the hidden state (and its combinations).
    for name in ("prefix", "hidden_raw", "prefix+hidden_raw", "prefix+rich+hidden_raw"):
        blocks = sets[name]
        x_train = np.concatenate([train[block] for block in blocks], axis=1).astype(np.float32)
        x_dev = np.concatenate([dev[block] for block in blocks], axis=1).astype(np.float32)
        x_test = np.concatenate([test[block] for block in blocks], axis=1).astype(np.float32)
        prediction, info = mlp_fit_predict(x_train, y_train, x_dev, y_dev, x_test)
        mse = float(((prediction - y_test) ** 2).mean())
        results[f"mlp_{name}"] = {
            "test_squared_error": mse,
            "n_features": int(x_train.shape[1]),
            **info,
        }
        print(
            f"mlp   {name:28s} features={x_train.shape[1]:4d} test_mse={mse:.4f} "
            f"(dev {info['dev_mse']:.4f}, wd {info['weight_decay']})",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("written", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
