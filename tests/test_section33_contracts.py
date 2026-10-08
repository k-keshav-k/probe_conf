"""Contract tests for the Section 3.3 implementation (CPU, fast).

Run with:  .venv/bin/python -m pytest tests/test_section33_contracts.py -q
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from section33.analysis.difficulty import within_question_auroc
from section33.analysis.evaluation import (
    bootstrap_p_value,
    calibration_guardrail_interval,
    decide_primary,
    decide_equivalence,
    decide_within_question,
    paired_subject_bootstrap,
)
from section33.analysis.interventions import (
    attenuate_topk,
    flatten_sharpen,
    identity,
    permuted_labels,
    random_matched,
    shuffle_within_strata,
)
from section33.analysis.power import (
    development_variance,
    paired_development_distribution,
    simulate_paired_decision,
)
from section33.data.cohort import (
    assign_popularity_strata,
    assign_splits,
    exclude_seen,
    validate_cohort,
    verify_disjoint,
)
from section33.data.labels import match_candidate, normalize, token_match
from section33.datasets import expand_by_multiplicity, load_split, metadata_features
from section33.features import families as fam
from section33.features import mlp_math as mm
from section33.features.hidden import DEFAULT_PROJ_LAYERS, HiddenProjector, proj_layers
from section33.features.probability import sequence_features
from section33.models.collect import ModelWeights, extract_features_for_sequences
from section33.models.contract import MLPCapture, mlp_modules, verify_mlp_contract
from section33.probes.calibration import corrected_calibration_error, fit_calibrator
from section33.probes.pipeline import Block, ProbeSpec, fit_probe, select_columns, standard_specs
from section33.smoke import _StubLoaded, _StubTokenizer, make_tiny_model


@pytest.fixture(scope="module")
def tiny_model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def tiny_loaded(tiny_model):
    return _StubLoaded(tiny_model, _StubTokenizer())


# ---------------------------------------------------------------------------
# MLP contract and invariants
# ---------------------------------------------------------------------------
def test_verify_mlp_contract(tiny_model):
    report = verify_mlp_contract(tiny_model)
    assert report["bias_free"], "the installed MLP must have no biases"
    assert report["reconstruction"]["max_abs_error"] < 1e-4


def test_hook_capture_matches_recomputation(tiny_model):
    capture = MLPCapture(tiny_model, [[4]], capture_output=True, keep_device=True)
    capture.register()
    with torch.no_grad():
        tiny_model(torch.tensor([[1, 2, 3, 4, 5]]), output_hidden_states=True, use_cache=False)
    capture.remove()
    gate, up, down = mlp_modules(tiny_model.model.layers[0])
    x = capture.captured[0]["x"][0, 0].float()
    a, b, s, o = mm.mlp_activations(x, gate.weight.float(), up.weight.float(), down.weight.float())
    assert torch.allclose(a, capture.captured[0]["a"][0, 0].float(), atol=1e-5)
    assert torch.allclose(s, capture.captured[0]["s"][0, 0].float(), atol=1e-5)
    assert torch.allclose(o, capture.captured[0]["o"][0, 0].float(), atol=1e-4)


def test_channel_rescaling_invariance(tiny_model):
    gate, up, down = mlp_modules(tiny_model.model.layers[0])
    x = torch.randn(gate.weight.shape[1])
    a, b, s, o = mm.mlp_activations(x, gate.weight.float(), up.weight.float(), down.weight.float())
    v_norms = down.weight.float().norm(dim=0)

    up_scaled = up.weight.float().clone()
    down_scaled = down.weight.float().clone()
    up_scaled[5] *= 3.0
    down_scaled[:, 5] /= 3.0
    b_scaled = up_scaled @ x
    s_scaled = torch.nn.functional.silu(a) * b_scaled
    o_scaled = down_scaled @ s_scaled

    before = mm.contribution_magnitudes(s, v_norms)
    after = mm.contribution_magnitudes(s_scaled, down_scaled.norm(dim=0))
    assert torch.allclose(before, after, atol=1e-4), "contribution mass must be scaling invariant"
    assert torch.allclose(o_scaled, o, atol=1e-4), "the MLP function must be scaling invariant"


def test_permutation_invariance(tiny_model):
    gate, up, down = mlp_modules(tiny_model.model.layers[0])
    x = torch.randn(gate.weight.shape[1])
    a, b, s, o = mm.mlp_activations(x, gate.weight.float(), up.weight.float(), down.weight.float())
    permutation = torch.randperm(down.weight.shape[1])
    up_perm = up.weight.float()[permutation, :]
    down_perm = down.weight.float()[:, permutation]
    s_perm = torch.nn.functional.silu(a)[permutation] * (up_perm @ x)
    o_perm = down_perm @ s_perm
    assert torch.allclose(o_perm, o, atol=1e-4)
    before = mm.concentration_summary(mm.contribution_magnitudes(s, down.weight.float().norm(dim=0)))
    after = mm.concentration_summary(mm.contribution_magnitudes(s_perm, down_perm.norm(dim=0)))
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-5), key


def test_reference_free_names(tiny_model):
    gate, up, down = mlp_modules(tiny_model.model.layers[0])
    x = torch.randn(gate.weight.shape[1])
    a, b, s, o = mm.mlp_activations(x, gate.weight.float(), up.weight.float(), down.weight.float())
    tensors = mm.MLPTensors(
        x=x, h_pre=x, h_post=x + o, a=a, b=b, s=s, o=o,
        v_norms=down.weight.float().norm(dim=0), w2_dot=x @ down.weight.float(),
    )
    names = fam.build_all(tensors)
    fam.assert_reference_free(names)
    assert not any("gold" in name or "margin" in name for name in names)


def test_extraction_covers_probability_and_projections(tiny_loaded):
    weights = ModelWeights(tiny_loaded.model, "cpu")
    projector = HiddenProjector(weights.n_layers, tiny_loaded.model.config.hidden_size, dim=4, seed=1)
    features = extract_features_for_sequences(
        tiny_loaded, weights, [[1, 2, 3, 4, 5]], [[4]], ["last_prompt"],
        hidden_projector=projector, probability=True,
    )[0]
    assert any(key.startswith("prob__top1__") for key in features)
    assert any(key.startswith("hiddenproj__post__") for key in features)
    assert any(key.startswith("contribution_concentration__") for key in features)


# ---------------------------------------------------------------------------
# Labels and cohort
# ---------------------------------------------------------------------------
def test_label_policy():
    assert match_candidate("Fred Trump was the father.", ["Fred Trump"]) == 1
    assert match_candidate("He was Fred Trump's father", ["Fred Trump"]) == 1
    assert match_candidate("Trump", ["Trump"]) == 1
    assert match_candidate("trumped", ["Trump"]) == 0, "short aliases need a whole-token match"
    assert normalize("The  United States!") == "united states"
    assert token_match("new york city", "new york")


def test_cohort_exclusion_quotas_and_splits():
    frame = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(40)],
            "subject_id": [f"s{i}" for i in range(40)],
            "relation": ["father"] * 20 + ["mother"] * 20,
            "popularity": list(range(20)) * 2,
        }
    )
    fresh = exclude_seen(frame, {"s0", "s1"})
    assert "s0" not in set(fresh["subject_id"])
    stratified = assign_popularity_strata(fresh)
    split = assign_splits(stratified, {"train": 10, "dev": 5, "calibration": 5, "test": 0}, seed=3)
    overlaps = verify_disjoint(split)
    assert all(value == 0 for value in overlaps.values())
    assert split.groupby("subject_id")["split"].nunique().max() == 1


# ---------------------------------------------------------------------------
# Probes and calibration
# ---------------------------------------------------------------------------
def _synthetic_frame(n=180, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(30), n // 30)
    signal = rng.normal(size=n)
    labels = (signal + rng.normal(size=n) * 0.8 > 0).astype(int)
    return pd.DataFrame(
        {
            "subject_id": groups,
            "label": labels,
            "f__contribution_concentration__e_norm_entropy__last_prompt__l0": signal,
            "f__gate_selection__s_pos_total__last_prompt__l0": -signal,
            "f__hiddenproj__post__last_prompt__l0__k0": signal * 0.5,
        }
    )


def test_preprocessing_fitted_inside_folds_only():
    frame = _synthetic_frame()
    column = "f__contribution_concentration__e_norm_entropy__last_prompt__l0"
    frame.loc[frame.index[-1], column] = 500.0
    spec = ProbeSpec("M", (Block("m", ("contribution_concentration__",), positions=("last_prompt",), pca_dim=2),))
    probe, _ = fit_probe(frame.iloc[:-1], frame["label"].iloc[:-1], frame["subject_id"].iloc[:-1], spec, seed=0)
    scaler_mean = float(probe.preprocessors["m"].scaler.mean_[0])
    train_mean = float(frame.iloc[:-1][column].to_numpy(dtype=np.float32).mean())
    assert abs(scaler_mean - train_mean) < 1e-4
    full_mean = float(frame[column].mean())
    assert abs(full_mean - scaler_mean) > 0.1, "the held-out outlier must not influence the scaler"


# ---------------------------------------------------------------------------
# Review repairs: feature selection, boundaries, joins, multiplicity, power
# ---------------------------------------------------------------------------
def test_probability_block_matches_position_suffix():
    specs = standard_specs()
    frame = pd.DataFrame(
        {
            "f__prob__top1__last_prompt": [0.1],
            "f__prob__entropy__answer_last_content": [0.2],
            "f__hiddenproj__post__last_prompt__l0__k0": [0.3],
        }
    )
    assert select_columns(frame, specs["P"].blocks[0]) == ["f__prob__top1__last_prompt"]
    assert select_columns(frame, specs["P_after"].blocks[0]) == ["f__prob__entropy__answer_last_content"]


def test_metadata_features_respect_track_boundary():
    frame = pd.DataFrame(
        {
            "track": ["A", "A"],
            "relation": ["father", "father"],
            "content_length": [1, 7],
            "popularity": [10.0, 10.0],
            "prompt_tokens": [5, 5],
        }
    )
    before = metadata_features(frame, track="A")
    assert "meta_content_length" not in before.columns
    assert before.loc[0].equals(before.loc[1]), "answer length must not change Track A inputs"
    after = metadata_features(frame, track="B")
    assert after["meta_content_length"].tolist() == [1.0, 7.0]


def test_duplicate_decoded_text_keeps_all_samples(tmp_path):
    common = dict(subject_id="s", question_id="q", split="test", track="B", multiplicity=4)
    labels = [
        {**common, "answer_index": 0, "candidate_text": "same", "label": 1},
        {**common, "answer_index": 1, "candidate_text": "same", "label": 0},
    ]
    features = [
        {
            **common,
            "answer_index": i,
            "candidate_token_ids": [i + 10],
            "position": "answer_last_content",
            "content_length": 1,
            "f__hiddenproj__post__answer_last_content__l0__k0": float(i),
        }
        for i in range(2)
    ]
    paths = {"labels": tmp_path / "labels", "features": tmp_path / "features"}
    for path in paths.values():
        path.mkdir()
    (paths["labels"] / "trackB_answers__stub.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels)
    )
    (paths["features"] / "trackB_features__stub.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in features)
    )
    joined = load_split(paths, "stub", "B", expected_samples=8)
    assert len(joined) == 2
    assert int(joined["multiplicity"].sum()) == 8


def test_join_rejects_unmatched_rows(tmp_path):
    common = dict(subject_id="s", question_id="q", split="test", track="B", multiplicity=4)
    labels = [
        {**common, "answer_index": 0, "candidate_text": "same", "label": 1},
        {**common, "answer_index": 1, "candidate_text": "same", "label": 0},
    ]
    features = [
        {
            **common,
            "answer_index": 0,
            "candidate_token_ids": [10],
            "position": "answer_last_content",
            "content_length": 1,
            "f__hiddenproj__post__answer_last_content__l0__k0": 0.0,
        }
    ]
    paths = {"labels": tmp_path / "labels", "features": tmp_path / "features"}
    for path in paths.values():
        path.mkdir()
    (paths["labels"] / "trackB_answers__stub.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels)
    )
    (paths["features"] / "trackB_features__stub.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in features)
    )
    with pytest.raises(ValueError, match="no matching feature"):
        load_split(paths, "stub", "B")


def test_paired_development_distribution_works_with_one_answer_subjects():
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "question_id": [f"q{i}" for i in range(n)],
            "label": rng.integers(0, 2, n),
            "score_a": rng.random(n),
            "score_b": rng.random(n),
            "relation": "father",
            "stratum": "0",
        }
    )
    distribution = paired_development_distribution(frame, "score_a", "score_b", n_boot=50, seed=0)
    assert distribution["n_subjects"] == n
    assert len(distribution["delta_auc"]) >= 2
    variance = development_variance(distribution)
    assert np.isfinite(variance["se_auc"]) and np.isfinite(variance["se_brier"])
    assert variance["n_subjects"] == n
    too_small = paired_development_distribution(
        frame.iloc[:1], "score_a", "score_b", n_boot=10, seed=0
    )
    assert not np.isfinite(development_variance(too_small)["se_auc"])


def test_power_scales_with_sample_size():
    # The earlier simulator drew population effects and reported <50% detection
    # at one million subjects for a large true effect.
    large = simulate_paired_decision(
        1_000_000, 0.040, 0.010, 1e-4, 1e-4, n_sim=2000, seed=0, calibration_ok_probability=1.0
    )
    assert large["p_useful_improvement_supported"] > 0.95
    null = simulate_paired_decision(
        1_000_000, 0.0, 0.0, 1e-4, 1e-4, n_sim=2000, seed=0, calibration_ok_probability=1.0
    )
    assert null["p_useful_improvement_supported"] < 0.01
    # A strongly negative effect with a precise Brier interval is classified by
    # the registered rules as "ruled out" rather than "harm".
    harmful = simulate_paired_decision(
        1_000_000, -0.100, -0.050, 1e-4, 1e-4, n_sim=2000, seed=0, calibration_ok_probability=1.0
    )
    assert harmful["p_useful_improvement_ruled_out"] > 0.95
    # Harm fires when the AUROC interval is below zero but the Brier interval
    # still crosses the useful margin.
    harm = simulate_paired_decision(
        1_000_000, -0.100, 0.020, 1e-4, 2e-2, n_sim=2000, seed=0, calibration_ok_probability=1.0
    )
    assert harm["p_harm_supported"] > 0.90


def test_validate_cohort_modes():
    partial = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "d"],
            "question_id": ["q1", "q2", "q3", "q4"],
            "relation": ["father", "father", "occupation", "occupation"],
            "stratum": ["0", "0", "0", "0"],
            "split": ["train", "train", "dev", "dev"],
        }
    )
    sizes = {"train": 2, "dev": 2, "calibration": 0, "test": 0}
    pilot = validate_cohort(partial, sizes, mode="pilot", excluded_subjects=0)
    assert not pilot["passed"]
    assert any("relation" in problem for problem in pilot["problems"])
    with pytest.raises(ValueError, match="confirmatory"):
        validate_cohort(partial, sizes, mode="confirmatory", excluded_subjects=0)

    complete = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "d"],
            "question_id": ["q1", "q2", "q3", "q4"],
            "relation": ["father", "mother", "place_of_birth", "occupation"],
            "stratum": ["0", "0", "0", "0"],
            "split": ["train"] * 4,
        }
    )
    good = validate_cohort(
        complete,
        {"train": 4, "dev": 0, "calibration": 0, "test": 0},
        mode="confirmatory",
        excluded_subjects=100,
    )
    assert good["passed"]


def test_simultaneous_confidence_widens_intervals():
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "label": rng.integers(0, 2, n),
            "a": rng.random(n),
            "b": rng.random(n),
            "relation": "father",
            "stratum": "0",
        }
    )
    narrow = paired_subject_bootstrap(frame, "a", "b", "label", n_boot=200, seed=0, confidence=0.95)
    wide = paired_subject_bootstrap(frame, "a", "b", "label", n_boot=200, seed=0, confidence=0.99)
    narrow_width = narrow["delta_auc_interval"][1] - narrow["delta_auc_interval"][0]
    wide_width = wide["delta_auc_interval"][1] - wide["delta_auc_interval"][0]
    assert wide_width >= narrow_width
    assert wide["confidence"] == 0.99


def test_corrected_calibration_error_cluster_aware():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1, 1, 1])
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    value = corrected_calibration_error(scores, labels, groups=groups)
    assert np.isfinite(value)


def test_flatten_sharpen_is_rescaling_invariant():
    s = torch.tensor([1.0, 1.0])
    w = torch.diag(torch.tensor([1.0, 10.0]))
    scale = torch.tensor([10.0, 1.0])
    normal = flatten_sharpen(s, w, exponent=0.5)
    rescaled = flatten_sharpen(s * scale, w / scale, exponent=0.5)
    assert torch.allclose(w @ s, (w / scale) @ (s * scale), atol=1e-6)
    assert torch.allclose(normal.o_new, rescaled.o_new, atol=1e-5)


def test_attenuate_removed_mass_accounts_for_partial_scaling():
    s = torch.tensor([1.0, 2.0])
    w = torch.eye(2)
    v_norms = w.norm(dim=0)
    result = attenuate_topk(s, w, v_norms, k=1, attenuation=0.5)
    assert result.removed_mass == pytest.approx(1.0)


def test_calibration_method_selection():
    frame = _synthetic_frame(seed=1)
    scores = np.clip(frame["f__hiddenproj__post__last_prompt__l0__k0"].to_numpy() * 0.2 + 0.5, 0, 1)
    calibrator, info = fit_calibrator(scores, frame["label"].to_numpy(), frame["subject_id"].to_numpy())
    assert info["selected"]["method"] in {"identity", "platt", "isotonic"}
    calibrated = calibrator.apply(scores)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


# ---------------------------------------------------------------------------
# Difficulty, evaluation, interventions
# ---------------------------------------------------------------------------
def test_within_question_auroc_half_ties():
    result = within_question_auroc(
        np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.4]),
        np.array([0, 1, 0, 1, 1, 0]),
        np.array(["q"] * 6),
    )
    assert result["within_question_auroc"] == pytest.approx(8.5 / 9)


def test_decision_rules():
    supported = decide_primary(
        {"delta_auc": 0.04, "delta_auc_interval": [0.025, 0.055],
         "delta_brier": 0.01, "delta_brier_interval": [0.006, 0.014]},
        calibration_error_increase=(0.001, 0.004),
    )
    assert supported.verdict == "useful_improvement_supported"
    ruled_out = decide_primary(
        {"delta_auc": 0.01, "delta_auc_interval": [0.002, 0.018],
         "delta_brier": 0.001, "delta_brier_interval": [-0.001, 0.003]},
        calibration_error_increase=(0.0, 0.004),
    )
    assert ruled_out.verdict == "useful_improvement_ruled_out"
    inconclusive = decide_primary(
        {"delta_auc": 0.02, "delta_auc_interval": [0.005, 0.035],
         "delta_brier": 0.006, "delta_brier_interval": [0.002, 0.010]},
        calibration_error_increase=(0.0, 0.004),
    )
    assert inconclusive.verdict == "inconclusive"
    assert decide_within_question((0.04, 0.08), (-0.01, 0.001)).verdict == "within_question_gain_supported"
    assert decide_equivalence((-0.01, 0.015)).verdict == "practical_equivalence"


def test_interventions_identity_and_matching():
    torch.manual_seed(0)
    w_down = torch.randn(6, 24)
    v_norms = w_down.norm(dim=0)
    s = torch.randn(24)
    ident = identity(s, w_down)
    assert torch.allclose(ident.o_new, w_down @ s, atol=1e-5)
    top = attenuate_topk(s, w_down, v_norms, k=3, attenuation=0.0)
    random = random_matched(
        s, w_down, v_norms, k=3,
        target_mass=top.removed_mass, target_norm=top.perturbation_norm, seed=2,
    )
    assert random.matched or "unidentifiable" in random.note


def test_negative_controls_keep_subject_blocks():
    labels = np.array([0, 1, 0, 1, 1, 0])
    subjects = np.array(["a", "a", "b", "b", "c", "c"])
    permuted = permuted_labels(labels, subjects, seed=0)
    for subject in ("a", "b", "c"):
        mask = subjects == subject
        assert sorted(permuted[mask].tolist()) == sorted(labels[mask].tolist())
    values = np.array([1.0, 2.0, 3.0, 4.0])
    strata = np.array(["x", "x", "y", "y"])
    shuffled = shuffle_within_strata(values, strata, seed=0)
    assert sorted(shuffled[:2].tolist()) == [1.0, 2.0]
    assert sorted(shuffled[2:].tolist()) == [3.0, 4.0]


# ---------------------------------------------------------------------------
# Regression tests for the review fixes
# ---------------------------------------------------------------------------
def test_sequence_features_score_every_candidate_token():
    """The off-by-one pairing scored token t+1 with the distribution for token t."""

    vocab = 4
    logits = torch.full((3, vocab), -20.0)
    for index in range(3):
        logits[index, index] = 20.0  # row t is confident on the correct token t
    out = sequence_features(logits, [0, 1, 2], "answer_last_content")
    assert out["prob__seq_logprob_sum__answer_last_content"] > -1e-3
    assert out["prob__seq_logprob_min__answer_last_content"] > -1e-3
    assert out["prob__seq_length__answer_last_content"] == 3.0
    one_token = sequence_features(logits[:1], [0], "answer_last_content")
    assert one_token["prob__seq_logprob_sum__answer_last_content"] > -1e-3


def test_family_tensor_and_scalar_agree(tiny_model):
    gate, up, down = mlp_modules(tiny_model.model.layers[0])
    x = torch.randn(gate.weight.shape[1])
    a, b, s, o = mm.mlp_activations(x, gate.weight.float(), up.weight.float(), down.weight.float())
    v_norms = down.weight.float().norm(dim=0)
    w2_dot = x @ down.weight.float()
    scalar_tensors = mm.MLPTensors(
        x=x, h_pre=x, h_post=x + o, a=a, b=b, s=s, o=o, v_norms=v_norms, w2_dot=w2_dot
    )
    batch_tensors = mm.MLPTensors(
        x=x.unsqueeze(0), h_pre=x.unsqueeze(0), h_post=(x + o).unsqueeze(0),
        a=a.unsqueeze(0), b=b.unsqueeze(0), s=s.unsqueeze(0), o=o.unsqueeze(0),
        v_norms=v_norms, w2_dot=w2_dot.unsqueeze(0),
    )
    scalar = fam.build_all(scalar_tensors)
    batched = fam.build_all_tensors(batch_tensors)
    assert set(scalar) == set(batched)
    for name, value in scalar.items():
        assert batched[name][0].item() == pytest.approx(value, rel=1e-5, abs=1e-6), name


def test_calibration_guardrail_direction():
    rng = np.random.default_rng(0)
    n = 240
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(60), 4).astype(str),
            "relation": "father",
            "stratum": "0",
            "label": (rng.random(n) > 0.5).astype(int),
        }
    )
    frame["overconfident"] = 0.9
    frame["calibrated"] = frame["label"] * 0.9 + 0.05
    low, high = calibration_guardrail_interval(
        frame, "overconfident", "calibrated", "label", n_boot=60, seed=0
    )
    assert low > 0, "first argument is worse calibrated, so the increase must be positive"
    reverse_low, _ = calibration_guardrail_interval(
        frame, "calibrated", "overconfident", "label", n_boot=60, seed=0
    )
    assert reverse_low < 0


def test_decision_requires_calibration_guardrail():
    result = {
        "delta_auc": 0.04,
        "delta_auc_interval": [0.025, 0.055],
        "delta_brier": 0.01,
        "delta_brier_interval": [0.006, 0.014],
    }
    assert decide_primary(result, calibration_error_increase=(0.0, 0.004)).verdict == "useful_improvement_supported"
    assert decide_primary(result, calibration_error_increase=(0.0, 0.020)).verdict != "useful_improvement_supported"


def test_metadata_excludes_stated_confidence():
    frame = pd.DataFrame(
        {
            "relation": ["father"],
            "stratum": ["0"],
            "content_length": [3],
            "conf": [90.0],
            "conf_parse_ok": [True],
        }
    )
    meta = metadata_features(frame)
    assert "meta_conf" not in meta.columns
    assert "conf_missing" in meta.columns
    specs = standard_specs()
    assert not any(
        "conf__" in prefix for block in specs["B"].blocks for prefix in block.include
    ), "the B baseline must not contain the stated confidence"
    assert any(
        "conf__" in prefix for block in specs["V"].blocks for prefix in block.include
    ), "the V comparator must exist"


def test_proj_layers_map_by_relative_depth():
    assert proj_layers(28) == DEFAULT_PROJ_LAYERS
    layers32 = proj_layers(32)
    assert len(layers32) == len(DEFAULT_PROJ_LAYERS)
    assert layers32[0] == 0 and layers32[-1] == 31
    assert all(0 <= layer < 32 for layer in layers32)
    assert proj_layers(16) == tuple(sorted({round(f * 15) for f in
                                            [layer / 27 for layer in DEFAULT_PROJ_LAYERS]}))


def test_expand_by_multiplicity():
    frame = pd.DataFrame({"multiplicity": [1, 3, 2], "value": [10, 20, 30]})
    expanded = expand_by_multiplicity(frame)
    assert len(expanded) == 6
    assert sorted(expanded["value"].tolist()) == [10, 20, 20, 20, 30, 30]
    assert len(expand_by_multiplicity(pd.DataFrame({"value": [1]}))) == 1


def _stub_candidate_rows(question_id, candidates):
    """Build label/feature rows for one Track B question.

    ``candidates`` is a list of ``(candidate_key, token_ids, text, multiplicity)``.
    """

    labels, features = [], []
    for answer_index, (key, tokens, text, multiplicity) in enumerate(candidates):
        common = {
            "model": "stub", "revision": "stub", "subject_id": "s1",
            "question_id": question_id, "split": "dev", "track": "B",
            "answer_index": answer_index, "generation_seed": 0,
        }
        labels.append(
            {
                **common, "candidate_key": key, "candidate_text": text, "label": 1,
                "label_policy_version": "test", "reference_answers": ["Fred"],
                "empty": False, "truncated": False, "refusal": False, "exception": None,
                "multiplicity": multiplicity, "relation": "father",
                "popularity": 1.0, "stratum": "0",
            }
        )
        features.append(
            {
                **common, "candidate_key": key, "candidate_token_ids": tokens,
                "position": "answer_last_content",
                "f__prob__top1__answer_last_content": 0.5,
                "prompt_token_hash": "x", "content_length": len(tokens),
                "feature_spec_version": "test", "multiplicity": multiplicity,
            }
        )
    return labels, features


def test_load_split_keeps_distinct_candidates_with_same_text(tmp_path):
    from section33.config import run_paths

    paths = run_paths(tmp_path)
    for folder in ("labels", "features"):
        (tmp_path / folder).mkdir()

    labels, features = [], []
    for question, candidates in {
        "q_same_key": [("k1", [1, 2], "Fred", 8)],
        "q_same_text": [("k2a", [1, 2], "same", 4), ("k2b", [3, 4], "same", 4)],
    }.items():
        block_labels, block_features = _stub_candidate_rows(question, candidates)
        labels += block_labels
        features += block_features
    (tmp_path / "labels" / "trackB_answers__stub__shard0of1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels)
    )
    (tmp_path / "features" / "trackB_features__stub__shard0of1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in features)
    )

    frame = load_split(paths, "stub", "B", expected_samples=8)
    assert len(frame) == 3, "distinct token sequences must not merge on decoded text"
    same_text = frame[frame["question_id"] == "q_same_text"]
    assert same_text["candidate_key"].nunique() == 2
    assert int(same_text["multiplicity"].sum()) == 8


def test_load_split_merges_same_candidate_across_shards(tmp_path):
    from section33.config import run_paths

    paths = run_paths(tmp_path)
    for folder in ("labels", "features"):
        (tmp_path / folder).mkdir()

    for tag in ("shard0of2", "shard1of2"):
        labels, features = _stub_candidate_rows("q1", [("k1", [1, 2], "Fred", 2)])
        (tmp_path / "labels" / f"trackB_answers__stub__{tag}.jsonl").write_text(
            json.dumps(labels[0]) + "\n"
        )
        (tmp_path / "features" / f"trackB_features__stub__{tag}.jsonl").write_text(
            json.dumps(features[0]) + "\n"
        )

    frame = load_split(paths, "stub", "B")
    assert len(frame) == 1
    assert int(frame["multiplicity"].iloc[0]) == 2


def test_bootstrap_p_value():
    assert bootstrap_p_value([0.1, 0.2, 0.3]) == pytest.approx(0.0)
    assert bootstrap_p_value([-0.1, -0.2]) == pytest.approx(0.0)
    assert bootstrap_p_value([-0.1, 0.1, 0.0]) == pytest.approx(1.0)


def test_cmd_evaluate_end_to_end(tmp_path, monkeypatch):
    """The frozen evaluation path runs, uses multiplicity, and writes every claim."""

    import dataclasses
    from types import SimpleNamespace

    import section33.analysis.evaluation as evaluation
    from section33.cli import cmd_evaluate, cmd_setup
    from section33.config import PROTOCOL, run_paths

    # Small bootstrap for test speed; production uses 10,000.
    monkeypatch.setattr(
        evaluation,
        "PROTOCOL",
        dataclasses.replace(PROTOCOL, bootstrap_samples=150, bootstrap_seed=1),
    )

    cmd_setup(SimpleNamespace(run=tmp_path))
    paths = run_paths(tmp_path)
    for folder in (tmp_path / "labels", tmp_path / "features", tmp_path / "evaluation"):
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    subjects = [f"s{index}" for index in range(60)]
    labels, confidences, features = [], [], []
    for index, subject in enumerate(subjects):
        if index < 36:
            split = "train"
        elif index < 48:
            split = "calibration"
        else:
            split = "test"
        ability = rng.normal()
        question = f"q{index}"
        for candidate in range(4):
            label = int(ability + rng.normal() * 0.8 > 0)
            confidence = float(np.clip(50 + 40 * ability + rng.normal() * 10, 0, 100))
            multiplicity = 5 if candidate == 0 else 1
            common = {
                "model": "stub", "revision": "stub", "subject_id": subject,
                "question_id": question, "split": split, "track": "B",
                "answer_index": candidate, "generation_seed": 0,
            }
            labels.append(
                {
                    **common, "candidate_text": f"answer {candidate}", "label": label,
                    "label_policy_version": "test", "reference_answers": ["answer"],
                    "empty": False, "truncated": False, "refusal": False, "exception": None,
                    "multiplicity": multiplicity, "relation": "father",
                    "popularity": 1.0, "stratum": "0",
                }
            )
            confidences.append(
                {
                    **common, "prompt_version": "test", "raw_text": str(int(confidence)),
                    "value": confidence, "parse_ok": True, "multiplicity": multiplicity,
                }
            )
            features.append(
                {
                    **common, "position": "answer_last_content",
                    "f__meta__rel_father": 1.0,
                    "f__meta__conf_missing": 0.0,
                    "f__prob__top1__answer_last_content": float(np.clip(0.5 + 0.2 * ability, 0, 1)),
                    "f__conf__stated": confidence,
                    "f__hiddenproj__post__answer_last_content__l0__k0": ability + rng.normal() * 0.3,
                    "f__contribution_concentration__e_top1_mass__answer_last_content__l0": ability + rng.normal() * 0.5,
                    "f__gate_selection__a_pos_total__answer_last_content__l0": ability + rng.normal() * 0.4,
                    "f__output_direction__o_out_norm__answer_last_content__l0": ability + rng.normal() * 0.4,
                    "f__w2_literal__dot_pos_total__answer_last_content__l0": ability + rng.normal() * 0.4,
                    "prompt_token_hash": "x", "candidate_token_ids": [candidate],
                    "content_length": 1, "feature_spec_version": "test",
                    "multiplicity": multiplicity,
                }
            )

    for name, rows in (
        ("trackB_answers__stub__shard0of1", labels),
        ("trackB_confidence__stub__shard0of1", confidences),
    ):
        with open(paths["labels"] / f"{name}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    with open(paths["features"] / "trackB_features__stub__shard0of1.jsonl", "w") as handle:
        for row in features:
            handle.write(json.dumps(row) + "\n")

    payload = cmd_evaluate(SimpleNamespace(run=tmp_path, model="stub", track="B", seed=0))

    assert payload["decision"]["verdict"] in {
        "useful_improvement_supported", "useful_improvement_ruled_out",
        "harm_supported", "inconclusive",
    }
    assert set(payload["secondary"]) == {
        "replication_hidden_plus_mlp_minus_hidden",
        "mlp_minus_hidden",
        "augmented_minus_confidence",
        "augmented_minus_probability",
    }
    assert payload["within_question"] is not None
    assert payload["within_question"]["decision"]["verdict"].startswith("within_question")
    assert payload["protocol_hash"] == PROTOCOL.content_hash()
    assert payload["feature_manifest"]["B_plus_M_after"]["total_columns"] > 0
    assert payload["difficulty"] is not None
    assert list((tmp_path / "probes").glob("fitted__*.json"))
    # Multiplicity is applied: 12 test subjects x 4 candidates with 5+1+1+1
    # draws each.
    assert payload["paired"]["n_subjects"] == 12
    assert payload["paired"]["n_observations"] == 12 * 8
    assert (tmp_path / "evaluation").glob("decision__*.json")
