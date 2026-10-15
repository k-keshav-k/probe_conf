"""CPU tests for the selective-metacognition toy package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from selective_metacognition.config import ToyConfig
from selective_metacognition.evaluation import (
    _squared_error_estimate,
    collect_report_rows,
    functional_table,
    reporting_metrics,
)
from selective_metacognition.model import (
    Selector,
    set_report_trainable,
    set_retrieval_trainable,
)
from selective_metacognition.runner import run_e1
from selective_metacognition.world import World


def tiny_config(**overrides) -> ToyConfig:
    values = dict(
        n_subjects=64,
        n_relations=4,
        n_answers=32,
        code_dim=16,
        memory_dim=64,
        train_subjects=48,
        dev_subjects=8,
        test_subjects=8,
        d_model=64,
        n_heads=4,
        d_hidden=128,
        n_layers=2,
        adapter_rank=4,
        adapter_layers=(1,),
        warm_start_steps=80,
        steps=15,
        batch=32,
        report_steps=15,
        rounds=1,
    )
    values.update(overrides)
    return ToyConfig(**values)


def test_world_conditions_have_known_semantics():
    config = tiny_config()
    world = World(config, seed=0, device="cpu")
    subjects = torch.arange(4)
    answers = world.assignments[subjects]
    memory = world.clean_memory(subjects)
    manual = sum(world.component(answers[:, relation], relation) for relation in range(config.n_relations))
    assert torch.allclose(memory, manual, atol=1e-5)

    weakened, correct = world.condition(subjects, 0, "target_weaken", strength=0.25)
    assert torch.equal(correct, answers[:, 0])
    expected = 0.75 * world.component(answers[:, 0], 0)
    assert torch.allclose(memory - weakened, expected, atol=1e-5)

    swapped, new_correct = world.condition(subjects, 0, "target_swap_valid", seed=3)
    assert not torch.equal(new_correct, answers[:, 0])
    assert torch.allclose(memory - swapped, world.component(answers[:, 0], 0) - world.component(new_correct, 0), atol=1e-5)

    false_memory, original = world.condition(subjects, 0, "false_memory", seed=3)
    assert torch.equal(original, answers[:, 0])
    assert torch.allclose(false_memory, swapped, atol=1e-5)

    irrelevant, _ = world.condition(subjects, 0, "irrelevant_swap", seed=3)
    difference = memory - irrelevant
    assert torch.allclose(difference[:, : config.code_dim], torch.zeros_like(difference[:, : config.code_dim]), atol=1e-5)


def test_adapter_starts_as_base_and_scopes_are_isolated():
    config = tiny_config()
    model = Selector(config)
    block = model.blocks[1]
    x = torch.randn(3, config.d_model)
    assert torch.allclose(block.gate(x), block.gate.base(x), atol=1e-6)
    assert torch.allclose(block.up(x), block.up.base(x), atol=1e-6)

    set_retrieval_trainable(model)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(names) == 4
    assert all("blocks.1.gate.lora_" in name or "blocks.1.up.lora_" in name for name in names)

    set_report_trainable(model)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert names
    assert all(name.startswith("report_head") for name in names)


def test_e1_smoke_runs_and_learns():
    config = tiny_config()
    report = run_e1(config, seeds=1, device="cpu")
    assert "aggregate" in report
    for arm in ("warm_start", "B", "C"):
        assert arm in report["aggregate"]
        clean_top1 = report["aggregate"][arm]["test"]["clean"]["top1"]
        assert clean_top1 > 1.0 / config.n_answers
    for split in ("train", "test", "fresh"):
        assert split in report["aggregate"]["C"]


def test_squared_error_estimate_formula():
    estimate = _squared_error_estimate(0.5, np.array([1.0, 0.0, 1.0, 0.0]))
    expected = 0.25 - 0.5 + (2.0 * 2.0 - 2.0) / (4 * 3)
    assert estimate == pytest.approx(expected)


def test_functional_table_and_report_rows_shapes():
    config = tiny_config()
    world = World(config, seed=1, device="cpu")
    model = Selector(config)
    table, per_item = functional_table(model, world, world.test_ids, seed_base=0)
    assert "selectivity" in table and "top1_share" in table
    assert per_item["clean"]["policy"].shape[0] == len(world.test_ids) * config.n_relations

    rows = collect_report_rows(model, world, world.test_ids, outcomes=4, seed=0)
    expected_rows = len(world.test_ids) * config.n_relations * len(
        ("clean", "target_weaken", "irrelevant_weaken", "irrelevant_swap", "target_swap_valid", "false_memory")
    )
    assert len(rows) == expected_rows


def test_incremental_brier_is_positive_for_informative_report():
    rng = np.random.default_rng(0)

    def make_rows(n: int) -> list[dict]:
        rows = []
        for index in range(n):
            p = float(rng.uniform(0.05, 0.95))
            outcomes = (rng.random(8) < p).astype(float).tolist()
            rows.append(
                {
                    "subject_id": index,
                    "relation": index % 4,
                    "condition": "clean",
                    "p_correct": p,
                    "report": p + 0.05 * float(rng.normal()),
                    "outcomes": outcomes,
                    "entropy": 1.0 + 0.1 * float(rng.normal()),
                    "max_prob": 0.5 + 0.1 * float(rng.normal()),
                    "margin": 0.1 + 0.05 * float(rng.normal()),
                }
            )
        return rows

    metrics = reporting_metrics(make_rows(400), make_rows(400), 4)
    assert metrics["incremental_brier"] > 0.02


def test_reporting_metrics_smoke():
    rows = []
    for index in range(48):
        rows.append(
            {
                "subject_id": index,
                "relation": index % 4,
                "condition": "clean",
                "p_correct": 0.5,
                "report": 0.6,
                "outcomes": [1.0, 0.0, 1.0, 1.0],
                "entropy": 1.0,
                "max_prob": 0.5,
                "margin": 0.1,
            }
        )
    metrics = reporting_metrics(rows, rows, 4)
    assert "brier_report" in metrics
    assert "incremental_brier" in metrics
    assert metrics["same_question"]["n"] == 0
