"""Runners for the staged toy experiments E1 and E2."""

from __future__ import annotations

import copy

import torch

from .config import ToyConfig
from .evaluation import (
    collect_report_rows,
    functional_table,
    reporting_metrics,
)
from .model import Selector
from .training import run_arm, train_retrieval, warm_start
from .world import World


def _mean(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).mean()) if values else float("nan")


def _aggregate(per_seed: dict[str, dict]) -> dict:
    """Mean of numeric leaves across seeds."""

    if not per_seed:
        return {}
    keys = list(next(iter(per_seed.values())).keys())
    aggregate = {}
    for key in keys:
        values = [seed_result[key] for seed_result in per_seed.values()]
        if isinstance(values[0], dict):
            aggregate[key] = _aggregate({str(index): value for index, value in enumerate(values)})
        elif isinstance(values[0], (int, float)):
            aggregate[key] = _mean([float(value) for value in values])
        else:
            aggregate[key] = values[0]
    return aggregate


def _device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_functional(model: Selector, world: World, seed: int) -> dict:
    fresh = world.resample_assignments(seed + 999)
    return {
        "train": functional_table(model, world, world.train_ids, seed_base=seed)[0],
        "test": functional_table(model, world, world.test_ids, seed_base=seed)[0],
        "fresh": functional_table(
            model, world, world.test_ids, assignments=fresh, seed_base=seed
        )[0],
    }


def run_e1(config: ToyConfig, seeds: int | None = None, device: torch.device | str | None = None) -> dict:
    """Arms B and C on the controlled toy (proposal section 10, E1)."""

    seeds = config.seeds if seeds is None else seeds
    device = _device(device)
    per_seed = {}
    for seed in range(seeds):
        world = World(config, seed=seed, device=device)
        model = Selector(config).to(device)
        warm_start(config, world, model, seed)
        results = {"warm_start": evaluate_functional(model, world, seed)}
        for arm in ("B", "C"):
            arm_model = copy.deepcopy(model)
            train_retrieval(config, world, arm_model, arm, seed)
            results[arm] = evaluate_functional(arm_model, world, seed)
        per_seed[f"seed_{seed}"] = results
    return {
        "config": config.as_dict(),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }


def run_e2(config: ToyConfig, seeds: int | None = None, device: torch.device | str | None = None) -> dict:
    """Arms A-D with alternating reporting training (proposal section 10, E2)."""

    seeds = config.seeds if seeds is None else seeds
    device = _device(device)
    per_seed = {}
    for seed in range(seeds):
        world = World(config, seed=seed, device=device)
        model = Selector(config).to(device)
        warm_start(config, world, model, seed)
        results = {}
        for arm in ("A", "B", "C", "D", "E"):
            arm_model = copy.deepcopy(model)
            run_arm(config, world, arm_model, arm, seed)
            functional = evaluate_functional(arm_model, world, seed)
            fresh = world.resample_assignments(seed + 999)
            train_rows = collect_report_rows(arm_model, world, world.train_ids, seed=seed)
            test_rows = collect_report_rows(arm_model, world, world.test_ids, seed=seed + 7)
            fresh_rows = collect_report_rows(
                arm_model, world, world.test_ids, assignments=fresh, seed=seed + 11
            )
            results[arm] = {
                "functional": functional,
                "reporting_test": reporting_metrics(train_rows, test_rows, config.n_relations),
                "reporting_fresh": reporting_metrics(train_rows, fresh_rows, config.n_relations),
            }
        per_seed[f"seed_{seed}"] = results
    return {
        "config": config.as_dict(),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }
