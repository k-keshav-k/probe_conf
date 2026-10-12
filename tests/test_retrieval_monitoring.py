"""CPU tests for the retrieval-monitoring package.

Covers the repaired S0 baselines, the S1 mask contract, relation
interventions, the endpoint, the toy world, the pilot analysis, and the
collector with a tiny model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from retrieval_monitoring.evaluation import paired_brier_endpoint
from retrieval_monitoring.interventions import (
    attenuate_state,
    attenuation_delta,
    fit_relation_map,
    matched_random_delta,
    projection,
)
from retrieval_monitoring.pilot import (
    PilotThresholds,
    select_subject_pairs,
    selectivity_report,
)
from retrieval_monitoring.retained_state import (
    additive_transform,
    continue_branch,
    full_run,
    generate_branch,
    locate_span,
    run_prefix,
)
from retrieval_monitoring.text_baselines import (
    evaluate_baseline,
    load_answers,
    load_questions,
    run_track,
)
from retrieval_monitoring.toy_world import ToyConfig, run_toy


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------
def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


class StubTokenizer:
    """Deterministic word-hash tokenizer for the tiny model (vocab 64)."""

    pad_token_id = 0
    eos_token_id = 1

    def _token(self, word: str) -> int:
        import re
        import zlib

        cleaned = re.sub(r"[^A-Za-z0-9]", "", word).lower() or "x"
        return 2 + (zlib.crc32(cleaned.encode()) % 60)

    def __call__(self, text, add_special_tokens=False):
        words = text.split()
        return {"input_ids": [self._token(word) for word in words]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in ids)


# ---------------------------------------------------------------------------
# S1: retained state
# ---------------------------------------------------------------------------
def test_retained_state_clean_and_padded_fork_equivalence():
    model = tiny_model()
    prefix = torch.tensor([[4, 5, 6, 7, 8, 9]])
    suffix = torch.tensor([[10, 11]])
    state = run_prefix(model, prefix)
    branch = continue_branch(model, state, suffix)
    reference = full_run(model, prefix, suffix)
    torch.testing.assert_close(
        branch.logits, reference.logits[:, -suffix.shape[1] :], atol=1e-5, rtol=1e-5
    )

    pad = 3
    padded = torch.full((1, prefix.shape[1] + pad), 3, dtype=torch.long)
    padded[:, pad:] = prefix
    mask = torch.zeros_like(padded)
    mask[:, pad:] = 1
    padded_state = run_prefix(model, padded, attention_mask=mask)
    padded_branch = continue_branch(model, padded_state, suffix)
    total_mask = torch.cat([mask, torch.ones_like(suffix)], dim=1)
    padded_full = full_run(model, padded, suffix, attention_mask=total_mask)
    torch.testing.assert_close(
        padded_branch.logits,
        padded_full.logits[:, -suffix.shape[1] :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_retained_state_intervention_and_generation():
    model = tiny_model()
    prefix = torch.tensor([[4, 5, 6, 7, 8, 9]])
    suffix = torch.tensor([[10, 11]])
    clean = run_prefix(model, prefix)
    clean_out = continue_branch(model, clean, suffix)
    vector = torch.zeros(model.config.hidden_size)
    vector[0] = 3.0
    intervened = run_prefix(
        model, prefix, layer_index=0, transform=additive_transform(vector, token_index=-1)
    )
    intervened_out = continue_branch(model, intervened, suffix)
    assert float((intervened_out.logits - clean_out.logits).abs().max()) > 1e-4
    last_layer = run_prefix(
        model,
        prefix,
        layer_index=model.config.num_hidden_layers - 1,
        transform=additive_transform(vector, token_index=-1),
    )
    last_out = continue_branch(model, last_layer, suffix)
    torch.testing.assert_close(last_out.logits, clean_out.logits, atol=1e-6, rtol=1e-5)
    generated = generate_branch(model, clean, suffix, max_new_tokens=4)
    assert generated.shape[1] == 4
    # A forked branch leaves the retained cache untouched.
    before = clean.cache.get_seq_length()
    continue_branch(model, clean, suffix)
    assert clean.cache.get_seq_length() == before


def test_state_check_runs_on_tiny_model():
    from retrieval_monitoring.checks import run_state_check

    class Loaded:
        pass

    loaded = Loaded()
    loaded.model = tiny_model()
    loaded.tokenizer = StubTokenizer()
    loaded.device = "cpu"
    checks = run_state_check(
        loaded,
        subject_text="Marie Curie",
        question="In what city was Marie Curie born?",
    )
    assert checks["passed"] is True
    assert checks["generated_tokens"] > 0
    assert checks["clean_replay_vs_full"] < checks["tolerance"]


def test_locate_span():
    tokenizer = StubTokenizer()
    ids = tokenizer("Subject: Marie Curie.", add_special_tokens=False)["input_ids"]
    start, end = locate_span(tokenizer, ids, "Marie Curie")
    assert end - start == 2
    assert ids[start:end] == tokenizer("Marie Curie", add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------
# S0: text baselines
# ---------------------------------------------------------------------------
def _synthetic_run(tmp_path: Path, track: str = "A", multiplicity: int = 1) -> Path:
    run = tmp_path
    (run / "labels").mkdir(parents=True, exist_ok=True)
    (run / "manifests").mkdir(parents=True, exist_ok=True)
    (run / "predictions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    answers = []
    for split, count in (("train", 60), ("dev", 20), ("test", 20)):
        for index in range(count):
            question_id = f"{split}-q{index}"
            subject_id = f"{split}-s{index}"
            label = int(rng.random() < 0.6)
            relation = "father" if index % 2 == 0 else "occupation"
            rows.append(
                {
                    "question_id": question_id,
                    "question": f"What is the answer for {split} {index}?",
                    "relation": relation,
                    "popularity": float(rng.integers(1, 100)),
                    "stratum": str(index % 5),
                    "split": split,
                    "subject_id": subject_id,
                    "subject_name": f"Person {index}",
                    "reference_answers": ["answer"],
                }
            )
            answers.append(
                {
                    "model": "stub",
                    "revision": "stub",
                    "subject_id": subject_id,
                    "question_id": question_id,
                    "split": split,
                    "track": track,
                    "answer_index": 0,
                    "generation_seed": 0,
                    "candidate_key": f"k{question_id}",
                    "candidate_text": f"answer {index}",
                    "label": label,
                    "label_policy_version": "test",
                    "reference_answers": ["answer"],
                    "empty": False,
                    "truncated": False,
                    "refusal": False,
                    "exception": None,
                    "multiplicity": multiplicity,
                    "relation": relation,
                    "popularity": float(rng.integers(1, 100)),
                    "stratum": str(index % 5),
                }
            )
    pd.DataFrame(rows).to_parquet(run / "manifests" / "cohort.parquet", index=False)
    with open(run / "labels" / f"track{track}_answers__stub.jsonl", "w") as handle:
        for row in answers:
            handle.write(json.dumps(row) + "\n")
    suffix = "_after" if track == "B" else ""
    predictions = pd.DataFrame(
        [
            {
                "question_id": row["question_id"],
                "subject_id": row["subject_id"],
                "split": "test",
                "label": answers[index]["label"],
                f"cal_H_512{suffix}": rng.random(),
                f"cal_P{suffix}": rng.random(),
                f"cal_V{suffix}": rng.random(),
                "relation": row["relation"],
                "popularity": row["popularity"],
                "stratum": row["stratum"],
                "candidate_text": answers[index]["candidate_text"],
            }
            for index, row in enumerate(rows)
            if row["split"] == "test"
        ]
    )
    predictions.to_parquet(
        run / "predictions" / f"test_predictions__qwen__qwen2.5_7b_instruct__track{track}.parquet",
        index=False,
    )
    return run


def test_text_baselines_validation_and_report(tmp_path):
    run = _synthetic_run(tmp_path, track="A")
    cohort = load_questions(run)
    answers = load_answers(run, "A", cohort)
    assert answers["question_id"].nunique() == len(cohort)
    reports, scores = run_track(run, "A", ("question",))
    report = reports[0]
    assert report.n_train_questions == 60
    assert report.n_dev_questions == 20
    assert report.n_test_questions == 20
    assert report.provenance["answer_files"]
    assert "text_score" in scores.columns

    # Wrong draw counts must be rejected.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "labels").mkdir()
    (bad / "manifests").mkdir()
    pd.DataFrame(
        [{"question_id": "q", "question": "x", "relation": "father", "split": "train",
          "subject_id": "s", "subject_name": "n", "reference_answers": ["a"],
          "popularity": 1.0, "stratum": "0"}]
    ).to_parquet(bad / "manifests" / "cohort.parquet", index=False)
    with open(bad / "labels" / "trackA_answers__stub.jsonl", "w") as handle:
        for _ in range(2):
            handle.write(json.dumps({"question_id": "q", "label": 1, "multiplicity": 1}) + "\n")
    with pytest.raises(ValueError, match="draw counts"):
        load_answers(bad, "A", load_questions(bad))


def test_text_baselines_track_b_draws(tmp_path):
    run = _synthetic_run(tmp_path, track="B", multiplicity=8)
    cohort = load_questions(run)
    answers = load_answers(run, "B", cohort, expected_draws=8)
    assert int(answers["multiplicity"].sum()) == 8 * len(cohort)
    reports, _ = run_track(run, "B", ("question", "question_candidate"))
    assert len(reports) == 2
    assert all(report.n_test_questions == 20 for report in reports)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------
def test_relation_map_selectivity_and_norm_matching():
    torch.manual_seed(0)
    dim, out_dim, n = 16, 8, 200
    states = torch.randn(n, dim)
    relation_a = torch.zeros(out_dim, dim)
    relation_a[:4, :4] = torch.randn(4, 4)  # answers depend on dims 0-3
    relation_b = torch.zeros(out_dim, dim)
    relation_b[:4, 4:8] = torch.randn(4, 4)  # answers depend on dims 4-7
    targets_a = states @ relation_a.T + 0.01 * torch.randn(n, out_dim)
    targets_b = states @ relation_b.T + 0.01 * torch.randn(n, out_dim)
    map_a = fit_relation_map(states, targets_a, "a", rank=4)
    map_b = fit_relation_map(states, targets_b, "b", rank=4)
    hidden = states[0] + torch.tensor([5.0] * 4 + [0.0] * (dim - 4))
    delta_a = attenuation_delta(hidden, map_a, alpha=0.5)
    delta_b = attenuation_delta(hidden, map_b, alpha=0.5)
    assert float(delta_a.norm()) > float(delta_b.norm()), "relevant map must bite harder"
    random_delta, _ = matched_random_delta(hidden, map_a, 0.5, seed=1)
    assert abs(float(random_delta.norm()) - float(delta_a.norm())) < 1e-5
    # Projection is idempotent.
    once = projection(hidden, map_a.basis)
    twice = projection(once, map_a.basis)
    assert float((once - twice).abs().max()) < 1e-5
    attenuated = attenuate_state(hidden, map_a, alpha=0.5)
    assert float((attenuated - hidden).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def _synthetic_endpoint_frame(informative: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for subject in range(80):
        split = "train" if subject < 40 else ("dev" if subject < 50 else "test")
        for question in range(2):
            base = rng.normal()
            for state in ("clean", "relevant", "other", "random"):
                shift = {"clean": 0.0, "relevant": -1.0, "other": -0.2, "random": -0.2}[state]
                p = 1.0 / (1.0 + np.exp(-(base + shift)))
                y = int(rng.random() < p)
                confidence = p if informative else 0.5
                rows.append(
                    {
                        "subject_id": f"s{subject}",
                        "question_id": f"q{subject}-{question}",
                        "state": state,
                        "split": split,
                        "baseline_score": float(np.clip(p + 0.3 * rng.normal(), 0.01, 0.99)),
                        "confidence": confidence + 0.01 * rng.normal(),
                        "y": y,
                    }
                )
    return pd.DataFrame(rows)


def test_endpoint_detects_information_and_shuffle():
    informative = _synthetic_endpoint_frame(True)
    result = paired_brier_endpoint(
        informative,
        baseline_columns=["baseline_score"],
        n_boot=100,
        n_shuffle=10,
    )
    assert result.improvement > 0.0
    assert result.shuffle_p_value is not None
    assert result.shuffle_p_value <= 0.2
    uninformative = _synthetic_endpoint_frame(False)
    flat = paired_brier_endpoint(
        uninformative,
        baseline_columns=["baseline_score"],
        n_boot=100,
        n_shuffle=10,
    )
    assert abs(flat.improvement) < 0.02


# ---------------------------------------------------------------------------
# Pilot analysis
# ---------------------------------------------------------------------------
def test_pilot_selectivity_report():
    rows = []
    rng = np.random.default_rng(0)
    for subject in range(40):
        for relation in ("father", "occupation"):
            for state, success in (
                ("clean", 0.95),
                ("relevant", 0.55),
                ("other", 0.90),
                ("random", 0.90),
            ):
                for draw in range(8):
                    rows.append(
                        {
                            "kind": "answer",
                            "subject_id": f"s{subject}",
                            "queried_relation": relation,
                            "state": state,
                            "draw": draw,
                            "label": int(rng.random() < success),
                        }
                    )
                rows.append(
                    {
                        "kind": "confidence",
                        "subject_id": f"s{subject}",
                        "queried_relation": relation,
                        "state": state,
                        "confidence": 80.0 if state == "clean" else 60.0,
                    }
                )
    report = selectivity_report(pd.DataFrame(rows), PilotThresholds(), n_boot=200)
    assert report["passed"] is True
    assert report["damage_relevant"]["mean"] > report["damage_other"]["mean"]
    assert report["confidence_diagnostic"]["relevant"]["mean"] > 0.0


def test_pilot_pair_selection_needs_two_relations():
    cohort = pd.DataFrame(
        [
            {"subject_id": "s1", "subject_name": "A", "relation": "father",
             "question_id": "q1", "question": "Q1?", "reference_answers": ["x"], "split": "dev"},
            {"subject_id": "s2", "subject_name": "B", "relation": "occupation",
             "question_id": "q2", "question": "Q2?", "reference_answers": ["y"], "split": "dev"},
        ]
    )
    with pytest.raises(ValueError, match="two questions per subject"):
        select_subject_pairs(cohort, 1, ("father", "occupation"))


# ---------------------------------------------------------------------------
# Toy world
# ---------------------------------------------------------------------------
def test_toy_world_smoke():
    config = ToyConfig(entities=128, dim=32, hidden=64, train_entities=96, heldout_entities=32,
                       steps=600, batch=64, seeds=1)
    results = run_toy(config)
    seed_zero = results["seed_0"]
    assert seed_zero["accuracy"] > 0.5, "the toy must be factually competent before interventions"
    assert seed_zero["relevant_success"] <= seed_zero["clean_success"] + 0.05


# ---------------------------------------------------------------------------
# Collector (tiny model, stub tokenizer)
# ---------------------------------------------------------------------------
def test_collector_rows_and_resume(tmp_path):
    from retrieval_monitoring.collector import (
        CollectorSettings,
        RetainedCollector,
        capture_subject_states,
    )
    from retrieval_monitoring.interventions import fit_relation_map
    from retrieval_monitoring.pilot import load_rows

    class Loaded:
        pass

    model = tiny_model()
    tokenizer = StubTokenizer()
    loaded = Loaded()
    loaded.model = model
    loaded.tokenizer = tokenizer
    loaded.device = "cpu"
    loaded.name = "stub"
    loaded.revision = "stub"

    subjects = [{"subject_id": "s1", "name": "Marie Curie"}]
    states = capture_subject_states(loaded, subjects, layer_index=0)
    # Fit maps from two synthetic states.
    state_tensor = torch.stack([states["s1"], states["s1"] + 0.1])
    targets = torch.randn(2, model.config.hidden_size)
    maps = {
        "father": fit_relation_map(state_tensor, targets, "father", rank=2),
        "occupation": fit_relation_map(state_tensor, targets, "occupation", rank=2),
    }
    pairs = [
        {
            "subject_id": "s1",
            "name": "Marie Curie",
            "relations": [
                {"relation": "father", "question_id": "q1", "question": "Who is the father?",
                 "reference_answers": ["Pierre"]},
                {"relation": "occupation", "question_id": "q2", "question": "What is the occupation?",
                 "reference_answers": ["physicist"]},
            ],
        }
    ]
    settings = CollectorSettings(layer_index=0, n_draws=1, max_new_tokens=2)
    out = tmp_path / "rows.jsonl"
    collector = RetainedCollector(loaded, settings, maps)
    first = collector.collect(pairs, states, out)
    rows = load_rows(out)
    assert first["rows_written"] > 0
    assert set(rows["kind"]) == {"answer", "confidence"}
    # A second run resumes and writes nothing new.
    second = collector.collect(pairs, states, out)
    assert second["rows_written"] == 0
