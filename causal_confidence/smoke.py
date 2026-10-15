"""CPU/small-model smoke checks for the causal-confidence pipeline.

Two levels:

- Fast checks (default): objective expectations versus exhaustive enumeration,
  policy-gradient agreement on a tiny categorical, Brier shapes, and retained
  state contract checks on a tiny random Qwen2 model.
- ``--end-to-end``: Qwen2.5-0.5B on a handful of audited questions, exercising
  P1 contracts and a reduced P2 discovery/validation.  The scientific gates can
  fail on the small model; the smoke asserts that the mechanics run and the
  artifacts are written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import interventions as iv
from . import stages
from .data import Question
from .episodes import run_prefix
from .evaluate import brier
from .objectives import (
    combined_expected_loss,
    enumerate_expected_loss,
    grid_moments,
)
from .protocol import CONFIDENCE_GRID, PROTOCOL

SMOKE_QUESTIONS = (
    ("In what city was Albert Einstein born?", ("Ulm",)),
    ("In what city was Marie Curie born?", ("Warsaw",)),
    ("In what city was Isaac Newton born?", ("Woolsthorpe", "Woolsthorpe-by-Colsterworth")),
    ("Who is the father of Prince William?", ("Charles III", "Charles", "King Charles III")),
    ("Who is the mother of Prince William?", ("Diana", "Diana, Princess of Wales")),
    ("Who is the father of Prince Harry?", ("Charles III", "Charles", "King Charles III")),
)


def smoke_questions() -> list[Question]:
    questions = []
    for index, (text, references) in enumerate(SMOKE_QUESTIONS):
        relation = (
            "place_of_birth"
            if "born" in text
            else ("father" if "father" in text else "mother")
        )
        questions.append(
            Question(
                question_id=f"smoke_{index}",
                subject_id=f"smoke_subject_{index // 2}",
                subject_name=text.split(" of ")[-1].rstrip("?") if " of " in text else text,
                relation=relation,
                question_text=text,
                references=tuple(references),
                source="smoke",
            )
        )
    return questions


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


def check_objectives() -> dict:
    torch.manual_seed(0)
    probs_a = torch.softmax(torch.randn(3, len(CONFIDENCE_GRID)), dim=-1)
    probs_b = torch.softmax(torch.randn(3, len(CONFIDENCE_GRID)), dim=-1)
    target_a = torch.rand(3)
    target_b = torch.rand(3)
    mu_a, var_a = grid_moments(probs_a)
    mu_b, var_b = grid_moments(probs_b)
    consensus = {}
    for index in range(3):
        expected = combined_expected_loss(
            mu_a[index : index + 1],
            var_a[index : index + 1],
            mu_b[index : index + 1],
            var_b[index : index + 1],
            target_a[index : index + 1],
            target_b[index : index + 1],
            lam=1.0,
        )
        exact = enumerate_expected_loss(
            probs_a[index : index + 1],
            probs_b[index : index + 1],
            target_a[index : index + 1],
            target_b[index : index + 1],
            lam=1.0,
        )
        consensus[index] = {
            "formula_total": float(expected["total"]),
            "enumeration_total": exact["total"],
        }
    gaps = [abs(item["formula_total"] - item["enumeration_total"]) for item in consensus.values()]
    return {"max_gap": max(gaps), "passed": max(gaps) < 1e-5}


def check_policy_gradient() -> dict:
    """Exact gradient versus the expected score-function estimator (tiny example)."""

    torch.manual_seed(0)
    logits = torch.tensor([0.4, -0.2, 0.1], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 0.3])

    def expected_reward(theta: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(theta, dim=-1) * rewards).sum()

    logits.grad = None
    expected_reward(logits).backward()
    exact = logits.grad.detach().clone()

    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=-1)
    estimator = torch.zeros_like(exact)
    for action in range(len(rewards)):
        onehot = torch.zeros_like(exact)
        onehot[action] = 1.0
        # d log p(a) / d theta = e_a - p
        estimator += probs[action] * rewards[action] * (onehot - probs)
    gap = float((estimator - exact).abs().max().item())
    return {"max_gap": gap, "passed": gap < 1e-6}


def check_brier_shapes() -> dict:
    scores = np.array([0.2, 0.8], dtype=np.float64)
    outcomes = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.float64)
    vector = brier(scores, outcomes)
    weights = np.array([3.0, 1.0])
    weighted = brier(scores, outcomes, weights=weights)
    expected = (
        ((scores[:, None] - outcomes) ** 2).mean(axis=1) * (weights / weights.sum())
    ).sum()
    return {
        "passed": abs(weighted - expected) < 1e-12,
        "value": vector,
    }


def check_episode_contract() -> dict:
    from .episodes import (
        check_prefix_immutability,
        duplicate_run_tolerance,
        last_layer_cache_miss,
    )

    model = tiny_model()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    suffix = torch.tensor([[12, 13]])
    positions = [[ids.shape[1] - 1]]
    group = iv.ChannelGroup(layer=1, channels=tuple(range(4)))
    tolerance = duplicate_run_tolerance(model, ids)
    floor = stages.cache_full_run_equivalence(
        model, ids, suffix, hook_specs=[], positions=[]
    )
    middle = stages.cache_full_run_equivalence(
        model,
        ids,
        suffix,
        hook_specs=[(group, "attenuate", 0.5)],
        positions=positions,
    )
    last = last_layer_cache_miss(
        model,
        ids,
        suffix,
        layer_index=model.config.num_hidden_layers - 1,
        positions=positions,
        channels=tuple(range(4)),
    )
    prefix = run_prefix(model, ids)
    isolation = check_prefix_immutability(model, prefix, suffix)
    tolerance_value = max(10 * floor["max_abs_diff"], 10 * tolerance["tolerance"], 1e-3)
    return {
        "middle_gap": middle["max_abs_diff"],
        "equivalence_tolerance": tolerance_value,
        "middle_passed": middle["max_abs_diff"] <= tolerance_value,
        "last_layer_gap": last["max_abs_diff"],
        "last_layer_misses_cache": not last["cache_reaches"],
        "isolation": isolation,
        "passed": middle["max_abs_diff"] <= tolerance_value
        and not last["cache_reaches"]
        and isolation["cache_unchanged"],
    }


def limit_cpu_threads(max_threads: int = 4) -> None:
    """Cap torch CPU threads; tiny MLP fits are dominated by thread overhead."""

    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    torch.set_num_threads(max(1, min(max_threads, os.cpu_count() or 1)))


def run_smoke(*, end_to_end: bool = False, device: str | None = None) -> int:
    limit_cpu_threads()
    results: dict = {}
    results["objectives"] = check_objectives()
    results["policy_gradient"] = check_policy_gradient()
    results["brier_shapes"] = check_brier_shapes()
    results["episode_contract"] = check_episode_contract()
    failures = [
        name
        for name in ("objectives", "policy_gradient", "brier_shapes", "episode_contract")
        if not results[name]["passed"]
    ]
    print(f"fast checks: {json_dumps(results)}")
    if end_to_end:
        results["end_to_end"] = run_end_to_end(device=device)
        print("end-to-end: " + json_dumps(results["end_to_end"]))
        if not results["end_to_end"]["mechanics_passed"]:
            failures.append("end_to_end")
    if failures:
        print(f"SMOKE FAILED: {failures}")
        return 1
    print("SMOKE OK")
    return 0


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


def run_end_to_end(*, device: str | None = None) -> dict:
    """Small-model P1+P2 mechanics run; scientific gates may fail by design."""

    import time

    from .cli import smoke_protocol
    from .protocol import ensure_run_dirs

    protocol = smoke_protocol(PROTOCOL)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path("runs/causal_confidence") / PROTOCOL.version / "smoke" / stamp
    paths = ensure_run_dirs(run_root)
    env = stages.load_model_and_tokenizer(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision=None,
        device=device,
        dtype="auto",
    )
    questions = smoke_questions()
    p1, banks = stages.run_p1(env, questions, draws=2)
    p1["smoke"] = True
    from .outcomes import write_banks, write_immutable

    write_immutable(paths["evaluation"] / "p1_episode_contract.json", p1)
    write_banks(paths["outcomes"] / "p1_discovery_banks.json", banks, stage="p1")

    discovery = stages.run_p2_discovery(
        env, questions, protocol=protocol, draws=2
    )
    write_immutable(paths["interventions"] / "p2_discovery.json", discovery)
    frozen = stages.FrozenIntervention.from_json(discovery["frozen"])
    validation, validation_banks = stages.run_p2_validation(
        env,
        questions,
        frozen=frozen,
        protocol=protocol,
        draws=2,
        random_replicates=1,
    )
    validation["smoke"] = True
    write_immutable(paths["interventions"] / "p2_validation.json", validation)
    write_banks(
        paths["outcomes"] / "p2_validation_banks.json",
        [bank for slots in validation_banks.values() for bank in slots.values()],
        stage="p2_validation",
    )
    stages.plot_p2_validation(validation, paths["figures"] / "p2_damage_restoration.png")
    stages.plot_p2_per_question(validation_banks, paths["figures"] / "p2_per_question.png")
    return {
        "contract_passed": p1["contract"]["passed"],
        "mean_answer_success": p1["answer_policy"]["mean_success"],
        "discovery_target": discovery["frozen"]["target"],
        "validation_gates": validation["gates"],
        "validation_passed": validation["passed"],
        "artifacts": str(paths["root"]),
        "mechanics_passed": bool(p1["contract"]["passed"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="causal-confidence smoke checks")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    raise SystemExit(
        run_smoke(end_to_end=arguments.end_to_end, device=arguments.device)
    )
