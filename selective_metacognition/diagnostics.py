"""Representation diagnostic for the prospective reporting failure.

Question: can *any* readout of the clean hidden state plus the intervention
description predict post-intervention success better than the baseline?

- If a probe (linear or MLP) beats the baseline on held-out subjects, the
  information is in the representation and the reporter training is the
  bottleneck.
- If no probe beats the baseline, the clean internal state does not carry
  item-specific fragility, which is a representational negative for the
  hypothesis as stated.

Probes are trained on the true post-intervention success probability (an upper
bound on what the representation makes readable) and, as a control, on the same
empirical outcome means the reporter sees.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import PROSPECTIVE_TYPES, ToyConfig
from .model import Selector
from .prospective import (
    ProspectiveSpec,
    apply_spec,
    intervention_context,
    training_specs,
)
from .training import _bindings, sample_batch, sample_outcomes, train_retrieval, warm_start
from .world import World

EPS = 1e-9


@torch.no_grad()
def collect_probe_data(
    config: ToyConfig,
    world: World,
    model: Selector,
    subject_ids: torch.Tensor,
    *,
    specs: list[ProspectiveSpec] | None = None,
    batches: int = 40,
    seed: int = 0,
) -> dict:
    """Clean hidden state, context, baseline features, true p, and outcomes."""

    specs = specs or training_specs()
    generator = torch.Generator().manual_seed(seed)
    hidden, context, p_post, outcomes, features = [], [], [], [], []
    for step in range(batches):
        subjects, relations = sample_batch(config, subject_ids, generator, world.device)
        bindings = _bindings(world, generator, step, 90_000)
        spec = specs[int(torch.randint(len(specs), (1,), generator=generator))]
        clean_memory, correct = world.condition(subjects, relations, "clean", assignments=bindings)
        post_memory, _ = apply_spec(world, subjects, relations, spec, bindings)
        clean_policy = model.policy(clean_memory, relations)
        post_policy = model.policy(post_memory, relations)
        p = post_policy.gather(1, correct[:, None]).squeeze(1)
        h = model.hidden(clean_memory, relations)
        ctx = (
            intervention_context(spec, world.device, clean_memory.dtype)
            .unsqueeze(0)
            .expand(len(subjects), -1)
        )
        top2 = clean_policy.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(clean_policy * (clean_policy + EPS).log()).sum(dim=1)
        y = sample_outcomes(post_policy, correct, config.outcomes_per_item, generator)

        relation_onehot = torch.zeros(len(subjects), config.n_relations, device=world.device)
        relation_onehot[torch.arange(len(subjects)), relations] = 1.0
        intervention_onehot = torch.zeros(len(subjects), len(PROSPECTIVE_TYPES), device=world.device)
        intervention_onehot[:, PROSPECTIVE_TYPES.index(spec.name)] = 1.0
        baseline = torch.cat(
            [
                relation_onehot,
                clean_policy.max(dim=1, keepdim=True).values,
                entropy[:, None],
                margin[:, None],
                intervention_onehot,
                ctx[:, -1:],
            ],
            dim=1,
        )
        hidden.append(h.cpu())
        context.append(ctx.cpu())
        p_post.append(p.cpu())
        outcomes.append(y.cpu())
        features.append(baseline.cpu())
    return {
        "hidden": torch.cat(hidden),
        "context": torch.cat(context),
        "p_post": torch.cat(p_post),
        "outcomes": torch.cat(outcomes),
        "features": torch.cat(features),
    }


def _train_mlp(x: np.ndarray, y: np.ndarray, seed: int, *, steps: int = 2000, hidden: int = 256) -> nn.Module:
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(x.shape[1], hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    for step in range(steps):
        index = torch.randint(len(x_tensor), (256,), generator=torch.Generator().manual_seed(step))
        loss = ((net(x_tensor[index]).squeeze(1) - y_tensor[index]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    return net


@torch.no_grad()
def _predict_mlp(net: nn.Module, x: np.ndarray) -> np.ndarray:
    return net(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()


def _brier_vs_outcomes(probability: np.ndarray, outcomes: np.ndarray) -> float:
    probability = np.clip(probability, 0.0, 1.0)[:, None]
    return float(np.mean((probability - outcomes) ** 2))


def _squared_error_vs_p(probability: np.ndarray, p_post: np.ndarray) -> float:
    return float(np.mean((np.clip(probability, 0.0, 1.0) - p_post) ** 2))


def run_probe_diagnostic(
    config: ToyConfig,
    arms: tuple[str, ...] = ("B", "C", "E"),
    *,
    seed: int = 0,
    batches: int = 40,
    device: torch.device | str | None = None,
) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge

    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    world = World(config, seed=seed, device=device)
    torch.manual_seed(seed)
    base_model = Selector(config).to(device)
    warm_start(config, world, base_model, seed)

    results: dict = {}
    for arm in arms:
        model = base_model
        if arm != "A":
            import copy

            model = copy.deepcopy(base_model)
            train_retrieval(config, world, model, arm, seed)

        train = collect_probe_data(config, world, model, world.train_ids, batches=batches, seed=seed)
        test = collect_probe_data(config, world, model, world.test_ids, batches=batches, seed=seed + 7)

        train_features = train["features"].numpy()
        test_features = test["features"].numpy()
        train_outcomes = train["outcomes"].numpy()
        test_outcomes = test["outcomes"].numpy()
        train_p = train["p_post"].numpy()
        test_p = test["p_post"].numpy()

        baseline = LogisticRegression(max_iter=2000, C=1.0).fit(
            np.repeat(train_features, train_outcomes.shape[1], axis=0),
            train_outcomes.reshape(-1),
        )
        baseline_prob = baseline.predict_proba(test_features)[:, 1]

        x_train = torch.cat([train["hidden"], train["context"]], dim=1).numpy()
        x_test = torch.cat([test["hidden"], test["context"]], dim=1).numpy()

        ridge = Ridge(alpha=1.0).fit(x_train, train_p)
        ridge_prob = ridge.predict(x_test)

        mlp_p = _train_mlp(x_train, train_p, seed)
        mlp_prob = _predict_mlp(mlp_p, x_test)

        mlp_outcomes = _train_mlp(x_train, train_outcomes.mean(axis=1), seed + 1)
        mlp_outcome_prob = _predict_mlp(mlp_outcomes, x_test)

        results[arm] = {
            "baseline": {
                "brier": _brier_vs_outcomes(baseline_prob, test_outcomes),
                "squared_error_vs_p": _squared_error_vs_p(baseline_prob, test_p),
            },
            "ridge_hidden_oracle": {
                "brier": _brier_vs_outcomes(ridge_prob, test_outcomes),
                "squared_error_vs_p": _squared_error_vs_p(ridge_prob, test_p),
            },
            "mlp_hidden_oracle": {
                "brier": _brier_vs_outcomes(mlp_prob, test_outcomes),
                "squared_error_vs_p": _squared_error_vs_p(mlp_prob, test_p),
            },
            "mlp_hidden_outcomes": {
                "brier": _brier_vs_outcomes(mlp_outcome_prob, test_outcomes),
                "squared_error_vs_p": _squared_error_vs_p(mlp_outcome_prob, test_p),
            },
        }
    return {"config": config.as_dict(), "seed": seed, "arms": results}
