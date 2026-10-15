"""P0: reporter positive control on demonstrably available signal.

Question (plan section 10, P0): can the reporting implementation learn a
signal that is demonstrably available?  We compare, on identical
train/development/test rows:

- the existing grid-mean head over the report grid,
- a direct scalar sigmoid regression,
- the successful probability baseline (logistic regression on outcome draws),

and separately train the heads to reproduce the baseline's predicted
probabilities.  Reproduction isolates representation learning from
correctness estimation.

Sources:

- ``synthetic``: a deterministic Gaussian design where the baseline is
  recoverable; used by smoke tests and unit checks.
- ``toy``: regenerate the minimum diagnostic data from the selective
  metacognition toy, because its checkpoints were not retained.

Gates: held-out MSE for reproducing baseline predictions at most 0.001, and a
same-input grid reporter no more than 0.002 Brier worse than the baseline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .baselines import fit_predict_outcomes
from .evaluate import brier
from .protocol import Gates, PROTOCOL
from .reporters import fit_grid_head, fit_scalar_head


@dataclass
class P0Rows:
    """Aligned arrays for the positive control."""

    features: np.ndarray
    hidden: np.ndarray
    outcomes: np.ndarray
    p_true: np.ndarray
    subjects: np.ndarray
    conditions: list[str] = field(default_factory=list)
    source: str = "synthetic"

    def __post_init__(self) -> None:
        n = len(self.features)
        for name in ("hidden", "outcomes", "p_true", "subjects"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")


def synthetic_rows(
    n_train: int = 4_000,
    n_test: int = 2_000,
    *,
    seed: int = 0,
    draws: int = 8,
    n_features: int = 6,
    hidden_dim: int = 32,
) -> P0Rows:
    """Gaussian design with a logistic success probability known to the evaluator."""

    generator = np.random.default_rng(seed)
    total = n_train + n_test
    features = generator.standard_normal((total, n_features)).astype(np.float32)
    weights = np.array([1.2, -0.8, 0.5, 0.35, -0.2, 0.15], dtype=np.float32)[:n_features]
    logit = features @ weights + 0.3 * np.sin(features[:, 0]) - 0.4
    p_true = 1.0 / (1.0 + np.exp(-0.7 * logit))
    outcomes = (generator.random((total, draws)) < p_true[:, None]).astype(np.float32)
    hidden = generator.standard_normal((total, hidden_dim)).astype(np.float32) * 0.1
    hidden += np.repeat(p_true[:, None], hidden_dim, axis=1) - 0.5
    subjects = np.array([f"s{index // 4}" for index in range(total)])
    conditions = ["clean"] * total
    return P0Rows(
        features=features,
        hidden=hidden,
        outcomes=outcomes,
        p_true=p_true,
        subjects=subjects,
        conditions=conditions,
        source="synthetic",
    )


def collect_toy_rows(
    *,
    arm: str = "B",
    seed: int = 0,
    batches: int = 40,
    device=None,
    smoke: bool = False,
    config=None,
) -> P0Rows:
    """Regenerate minimum diagnostic rows from the selective-metacognition toy.

    The toy's checkpoints were not retained, so this reruns the warm start and
    one retrieval arm with a documented reduced configuration in smoke mode.
    """

    import torch

    from selective_metacognition.config import REPORT_CONDITIONS, ToyConfig
    from selective_metacognition.model import Selector
    from selective_metacognition.training import (
        _bindings,
        sample_batch,
        sample_outcomes,
        train_retrieval,
        warm_start,
    )
    from selective_metacognition.world import World

    if config is None:
        config = ToyConfig()
    if smoke:
        config = dataclasses.replace(
            config,
            warm_start_steps=40,
            steps=30,
            batch=32,
            train_subjects=48,
            dev_subjects=8,
            test_subjects=8,
            n_subjects=64,
            n_relations=2,
            n_answers=32,
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    world = World(config, seed=seed, device=device)
    torch.manual_seed(seed)
    model = Selector(config).to(device)
    warm_start(config, world, model, seed)
    if arm != "A":
        model = train_retrieval(config, world, model, arm, seed)

    def collect(subject_ids, tag: int) -> P0Rows:
        generator = torch.Generator().manual_seed(seed + tag)
        features, hidden, outcomes, p_true, subjects, conditions = [], [], [], [], [], []
        with torch.no_grad():
            for step in range(batches):
                subject_batch, relations = sample_batch(
                    config, subject_ids, generator, world.device
                )
                assignments = _bindings(world, generator, step, 30_000 + tag)
                name = REPORT_CONDITIONS[
                    int(torch.randint(len(REPORT_CONDITIONS), (1,), generator=generator))
                ]
                memory, correct = world.condition(
                    subject_batch,
                    relations,
                    name,
                    seed=int(generator.initial_seed()) + step,
                    assignments=assignments,
                )
                policy = model.policy(memory, relations)
                p = policy.gather(1, correct[:, None]).squeeze(1)
                h = model.hidden(memory, relations)
                y = sample_outcomes(policy, correct, config.outcomes_per_item, generator)
                onehot = torch.zeros(
                    len(subject_batch), config.n_relations, device=world.device
                )
                onehot[torch.arange(len(subject_batch)), relations] = 1.0
                top2 = policy.topk(2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                entropy = -(policy * (policy + 1e-9).log()).sum(dim=1)
                base = torch.cat(
                    [onehot, policy.max(dim=1, keepdim=True).values, entropy[:, None], margin[:, None]],
                    dim=1,
                )
                features.append(base.cpu().numpy())
                hidden.append(h.cpu().numpy())
                outcomes.append(y.cpu().numpy())
                p_true.append(p.cpu().numpy())
                subjects.extend(str(int(s)) for s in subject_batch.cpu())
                conditions.append(name)
        return P0Rows(
            features=np.concatenate(features).astype(np.float32),
            hidden=np.concatenate(hidden).astype(np.float32),
            outcomes=np.concatenate(outcomes).astype(np.float32),
            p_true=np.concatenate(p_true).astype(np.float32),
            subjects=np.asarray(subjects),
            conditions=conditions,
            source=f"toy_{arm}",
        )

    train = collect(world.train_ids, 0)
    test = collect(world.test_ids, 100)
    return P0Rows(
        features=np.concatenate([train.features, test.features]),
        hidden=np.concatenate([train.hidden, test.hidden]),
        outcomes=np.concatenate([train.outcomes, test.outcomes]),
        p_true=np.concatenate([train.p_true, test.p_true]),
        subjects=np.concatenate([train.subjects, test.subjects]),
        conditions=train.conditions + test.conditions,
        source=f"toy_{arm}",
    )


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2))


def run_p0_controls(
    rows: P0Rows,
    *,
    train_fraction: float = 0.67,
    seed: int = 0,
    gates: Gates = PROTOCOL.gates,
    head_steps: int = 1_500,
    head_hidden: int = 256,
) -> dict:
    """Train the head comparisons and evaluate the P0 gates.

    Rows are split by subject so the held-out set never shares a subject with
    training.  The heads and the baseline see identical feature matrices.
    """

    import torch

    subjects = list(dict.fromkeys(rows.subjects.tolist()))
    generator = np.random.default_rng(seed)
    generator.shuffle(subjects)
    n_train_subjects = max(1, int(len(subjects) * train_fraction))
    train_subjects = set(subjects[:n_train_subjects])
    train_mask = np.array([subject in train_subjects for subject in rows.subjects])
    test_mask = ~train_mask
    if not test_mask.any():
        raise ValueError("no held-out subjects in the P0 split")

    x_train = torch.tensor(rows.features[train_mask], dtype=torch.float32)
    x_test = torch.tensor(rows.features[test_mask], dtype=torch.float32)
    h_train = torch.tensor(rows.hidden[train_mask], dtype=torch.float32)
    h_test = torch.tensor(rows.hidden[test_mask], dtype=torch.float32)
    y_train = rows.outcomes[train_mask]
    y_test = rows.outcomes[test_mask]
    p_train = rows.p_true[train_mask]
    p_test = rows.p_true[test_mask]

    # Probability baseline on the same inputs.
    baseline_test = fit_predict_outcomes(
        rows.features[train_mask], y_train, rows.features[test_mask], kind="logistic", seed=seed
    )
    baseline_train = fit_predict_outcomes(
        rows.features[train_mask], y_train, rows.features[train_mask], kind="logistic", seed=seed
    )
    baseline_metrics = {
        "brier": brier(baseline_test, y_test),
        "squared_error_vs_p": _mse(baseline_test, p_test),
    }

    def train_heads(input_train, input_test, name_prefix: str) -> dict:
        result: dict = {}
        grid = fit_grid_head(
            input_train,
            torch.tensor(y_train.mean(axis=1), dtype=torch.float32),
            seed=seed,
            steps=head_steps,
            hidden=head_hidden,
        )
        with torch.no_grad():
            grid_test = grid.expected(input_test).numpy()
        result[f"{name_prefix}_grid"] = {
            "brier": brier(grid_test, y_test),
            "squared_error_vs_p": _mse(grid_test, p_test),
            "brier_array": ((np.clip(grid_test, 0, 1)[:, None] - y_test) ** 2).mean(axis=1),
        }
        scalar = fit_scalar_head(
            input_train,
            torch.tensor(y_train.mean(axis=1), dtype=torch.float32),
            seed=seed,
            steps=head_steps,
            hidden=head_hidden,
        )
        with torch.no_grad():
            scalar_test = scalar(input_test).numpy()
        result[f"{name_prefix}_scalar"] = {
            "brier": brier(scalar_test, y_test),
            "squared_error_vs_p": _mse(scalar_test, p_test),
        }
        return result

    correctness = train_heads(x_train, x_test, "features")
    hidden_only = train_heads(h_train, h_test, "hidden")
    combined = train_heads(
        torch.cat([x_train, h_train], dim=1),
        torch.cat([x_test, h_test], dim=1),
        "hidden_plus_features",
    )

    # Reproduction of the baseline's predicted probabilities.
    baseline_train_tensor = torch.tensor(baseline_train, dtype=torch.float32)
    baseline_test_tensor = torch.tensor(baseline_test, dtype=torch.float32)
    repro_grid = fit_grid_head(
        x_train, baseline_train_tensor, seed=seed, steps=head_steps, hidden=head_hidden
    )
    repro_scalar = fit_scalar_head(
        x_train, baseline_train_tensor, seed=seed, steps=head_steps, hidden=head_hidden
    )
    with torch.no_grad():
        repro_grid_test = repro_grid.expected(x_test).numpy()
        repro_scalar_test = repro_scalar(x_test).numpy()
    reproduction = {
        "grid_mse": _mse(repro_grid_test, baseline_test),
        "scalar_mse": _mse(repro_scalar_test, baseline_test),
    }

    # Oracle targets: the true success probability from the world.
    oracle_grid = fit_grid_head(
        x_train, torch.tensor(p_train, dtype=torch.float32), seed=seed, steps=head_steps, hidden=head_hidden
    )
    with torch.no_grad():
        oracle_test = oracle_grid.expected(x_test).numpy()
    oracle = {
        "brier": brier(oracle_test, y_test),
        "squared_error_vs_p": _mse(oracle_test, p_test),
    }

    grid_brier = correctness["features_grid"]["brier"]
    gate_reproduction = {
        "value": reproduction["grid_mse"],
        "threshold": gates.p0_reproduction_mse,
        "passed": reproduction["grid_mse"] <= gates.p0_reproduction_mse,
    }
    gate_correctness = {
        "reporter_brier": grid_brier,
        "baseline_brier": baseline_metrics["brier"],
        "difference": grid_brier - baseline_metrics["brier"],
        "tolerance": gates.p0_brier_tolerance,
        "passed": grid_brier - baseline_metrics["brier"] <= gates.p0_brier_tolerance,
    }
    failure_analysis = None
    if not (gate_reproduction["passed"] and gate_correctness["passed"]):
        parts = []
        if not gate_reproduction["passed"]:
            parts.append(
                "the grid head did not reproduce the baseline's predictions to the "
                "0.001 tolerance; representation learning is unresolved"
            )
        if not gate_correctness["passed"]:
            parts.append(
                "the same-input grid reporter trails the probability baseline by more "
                "than 0.002 Brier; investigate regularization/data differences before "
                "interpreting later reporter failures"
            )
        failure_analysis = "; ".join(parts)

    return {
        "source": rows.source,
        "n_rows": int(len(rows.features)),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "n_features": int(rows.features.shape[1]),
        "baseline": baseline_metrics,
        "correctness_heads": correctness,
        "hidden_only": hidden_only,
        "hidden_plus_features": combined,
        "reproduction": reproduction,
        "oracle_probability": oracle,
        "gates": {
            "reproduction_mse": gate_reproduction,
            "same_input_brier": gate_correctness,
        },
        "passed": bool(gate_reproduction["passed"] and gate_correctness["passed"]),
        "failure_analysis": failure_analysis,
        "config": {
            "seed": seed,
            "train_fraction": train_fraction,
            "head_steps": head_steps,
            "head_hidden": head_hidden,
        },
    }
