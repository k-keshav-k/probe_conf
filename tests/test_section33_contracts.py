"""Contract tests for the Section 3.3 implementation (CPU, fast).

Run with:  .venv/bin/python -m pytest tests/test_section33_contracts.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from section33.analysis.difficulty import within_question_auroc
from section33.analysis.evaluation import decide_primary, decide_equivalence, decide_within_question
from section33.analysis.interventions import (
    attenuate_topk,
    identity,
    permuted_labels,
    random_matched,
    shuffle_within_strata,
)
from section33.data.cohort import assign_popularity_strata, assign_splits, exclude_seen, verify_disjoint
from section33.data.labels import match_candidate, normalize, token_match
from section33.features import families as fam
from section33.features import mlp_math as mm
from section33.features.hidden import HiddenProjector
from section33.models.collect import ModelWeights, extract_features_for_sequences
from section33.models.contract import MLPCapture, mlp_modules, verify_mlp_contract
from section33.probes.calibration import fit_calibrator
from section33.probes.pipeline import Block, ProbeSpec, fit_probe
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
