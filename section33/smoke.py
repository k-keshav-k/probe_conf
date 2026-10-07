"""Smoke checks for the Section 3.3 implementation.

Three levels, safe to run on a laptop:

1. ``smoke_contracts``  — tiny random Qwen2 model on CPU: hook identity,
   reconstruction, channel-rescaling invariance, permutation invariance, and
   reference-free feature naming.
2. ``smoke_pipeline``   — synthetic data: fold-internal preprocessing, probe
   selection, calibration, within-question AUROC, interventions, decision rules.
3. ``smoke_end_to_end`` — optional small model (default Qwen2.5-0.5B-Instruct):
   two questions through Track A and Track B, then check the artefact files.

Usage:
    python -m section33.smoke                  # levels 1 and 2 (CPU only)
    python -m section33.smoke --end-to-end     # also level 3
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL, run_paths
from .features import families as fam
from .features import mlp_math as mm
from .models.contract import MLPCapture, mlp_modules, verify_mlp_contract
from .utils import log


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class _StubLoaded:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.name = "stub"
        self.revision = "stub"
        self.device = str(next(model.parameters()).device)
        self.dtype = "float32"


def make_tiny_model(seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config).eval()
    return model


# ---------------------------------------------------------------------------
# Level 1: contracts
# ---------------------------------------------------------------------------
def smoke_contracts() -> dict:
    results: dict[str, object] = {}
    model = make_tiny_model()
    results["verify_mlp_contract"] = verify_mlp_contract(model)

    # Capture activations for one sequence.
    sequence = [1, 2, 3, 4, 5]
    capture = MLPCapture(model, [[4]], capture_output=True, keep_device=True)
    capture.register()
    with torch.no_grad():
        model(torch.tensor([sequence]), output_hidden_states=True, use_cache=False)
    capture.remove()

    layer = 0
    gate, up, down = mlp_modules(model.model.layers[layer])
    x = capture.captured[layer]["x"][0, 0].float()
    a = capture.captured[layer]["a"][0, 0].float()
    b = capture.captured[layer]["b"][0, 0].float()
    s = capture.captured[layer]["s"][0, 0].float()
    o = capture.captured[layer]["o"][0, 0].float()
    a_ref, b_ref, s_ref, o_ref = mm.mlp_activations(
        x, gate.weight.float(), up.weight.float(), down.weight.float()
    )
    errors = {
        "a": float((a - a_ref).abs().max()),
        "b": float((b - b_ref).abs().max()),
        "s": float((s - s_ref).abs().max()),
        "o": float((o - o_ref).abs().max()),
    }
    results["hook_reconstruction"] = {"errors": errors, "passed": max(errors.values()) < 1e-4}

    # Channel rescaling invariance of contribution features.
    v_norms = down.weight.float().norm(dim=0)
    up_scaled = up.weight.float().clone()
    down_scaled = down.weight.float().clone()
    channel = 3
    factor = 2.5
    up_scaled[channel, :] *= factor
    down_scaled[:, channel] /= factor
    b_scaled = up_scaled @ x
    s_scaled = torch.nn.functional.silu(a) * b_scaled
    o_scaled = down_scaled @ s_scaled
    contribution_after = mm.contribution_magnitudes(s_scaled, down_scaled.norm(dim=0))
    contribution_before = mm.contribution_magnitudes(s, v_norms)
    results["rescale_contribution_error"] = float((contribution_after - contribution_before).abs().max())
    results["rescale_output_error"] = float((o_scaled - down.weight.float() @ s).abs().max())
    results["rescale_passed"] = results["rescale_contribution_error"] < 1e-4 and results["rescale_output_error"] < 1e-4

    # Permutation invariance: shuffle channels with weights.
    perm = torch.randperm(down.weight.shape[1])
    up_perm = up.weight.float()[perm, :]
    down_perm = down.weight.float()[:, perm]
    s_perm = torch.nn.functional.silu(a)[perm] * (up_perm @ x)
    o_perm = down_perm @ s_perm
    summaries_before = mm.concentration_summary(mm.contribution_magnitudes(s, v_norms))
    summaries_after = mm.concentration_summary(mm.contribution_magnitudes(s_perm, down_perm.norm(dim=0)))
    order_errors = {
        key: float((summaries_before[key] - summaries_after[key]).abs())
        for key in summaries_before
    }
    results["permutation_summary_error"] = order_errors
    results["permutation_output_error"] = float((o_perm - down.weight.float() @ s).abs().max())
    results["permutation_passed"] = max(order_errors.values()) < 1e-5 and results["permutation_output_error"] < 1e-4

    # Reference-free feature names.
    tensors = mm.MLPTensors(
        x=x, h_pre=x, h_post=x + o, a=a, b=b, s=s, o=o,
        v_norms=v_norms, w2_dot=x @ down.weight.float(),
    )
    names = fam.build_all(tensors).keys()
    try:
        fam.assert_reference_free(names)
        results["reference_free"] = {"passed": True}
    except ValueError as error:
        results["reference_free"] = {"passed": False, "error": str(error)}

    results["passed"] = all(
        bool(results[key])
        for key in (
            "rescale_passed",
            "permutation_passed",
        )
    ) and bool(results["verify_mlp_contract"]["passed"]) and bool(
        results["hook_reconstruction"]["passed"]
    ) and bool(results["reference_free"]["passed"])
    return results


def _contribution_features(x, s, v_norms, w_down):
    tensors = mm.MLPTensors(
        x=x,
        h_pre=x,
        h_post=x,
        a=torch.zeros_like(s),
        b=torch.zeros_like(s),
        s=s,
        o=w_down @ s,
        v_norms=v_norms,
        w2_dot=None,
    )
    return fam.build_family("contribution_concentration", tensors)


# ---------------------------------------------------------------------------
# Level 2: pipeline on synthetic data
# ---------------------------------------------------------------------------
def smoke_pipeline() -> dict:
    import pandas as pd

    from .analysis.difficulty import within_question_auroc
    from .analysis.evaluation import decide_primary, paired_subject_bootstrap
    from .analysis.interventions import attenuate_topk, identity, random_matched
    from .probes.calibration import fit_calibrator
    from .probes.pipeline import Block, ProbeSpec, fit_probe

    rng = np.random.default_rng(0)
    n = 240
    groups = np.repeat(np.arange(40), 6)
    signal = rng.normal(size=n)
    label = (signal + rng.normal(size=n) * 0.8 > 0).astype(int)
    frame = pd.DataFrame(
        {
            "subject_id": groups,
            "f__contribution_concentration__e_norm_entropy__last_prompt__l0": signal,
            "f__contribution_concentration__e_top1_mass__last_prompt__l0": -signal,
            "f__hidden__x__last_prompt__l0": signal * 0.5,
        }
    )
    spec = ProbeSpec(
        "M",
        (Block("m", ("contribution_concentration__",), positions=("last_prompt",)),),
    )
    probe, best = fit_probe(frame, label, groups, spec, seed=0)
    prediction = probe.predict(
        {block.name: frame[[c for c in frame.columns if block.matches(c)]].to_numpy(np.float32)
         for block in spec.blocks}
    )
    results: dict[str, object] = {
        "fit_probe": {
            "config": best["config"],
            "mean_brier": best["brier"],
            "prediction_range": [float(prediction.min()), float(prediction.max())],
        }
    }

    # Fold-internal preprocessing check: a test row far outside the training
    # range must be transformed with training statistics only.
    column = "f__contribution_concentration__e_norm_entropy__last_prompt__l0"
    frame2 = frame.copy()
    frame2.loc[frame2.index[-1], column] = 100.0
    spec2 = ProbeSpec("M", (Block("m", ("contribution_concentration__",), positions=("last_prompt",), pca_dim=1),))
    probe2, _ = fit_probe(frame2.iloc[:-1], label[:-1], groups[:-1], spec2, seed=0)
    scaler_mean = float(probe2.preprocessors["m"].scaler.mean_[0])
    train_mean = float(np.mean(frame2.iloc[:-1][column].to_numpy(dtype=np.float32)))
    full_mean = float(np.mean(frame2[column].to_numpy(dtype=np.float64)))
    results["preprocessing_train_only"] = {
        "scaler_mean": scaler_mean,
        "train_mean": train_mean,
        "full_mean_including_outlier": full_mean,
        "passed": abs(scaler_mean - train_mean) < 1e-4 and abs(full_mean - scaler_mean) > 0.1,
    }

    # Calibration selection runs without labels leaking across folds.
    scores = np.clip(prediction * 0.6 + 0.2, 0, 1)
    calibrator, calibration_info = fit_calibrator(scores, label, groups)
    results["calibration"] = {
        "selected": calibration_info["selected"]["method"],
        "applied_range": [float(calibrator.apply(scores).min()), float(calibrator.apply(scores).max())],
    }

    # Within-question AUROC with a tie.
    within = within_question_auroc(
        np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.4]),
        np.array([0, 1, 0, 1, 1, 0]),
        np.array(["q", "q", "q", "q", "q", "q"]),
    )
    results["within_question"] = {"value": within["within_question_auroc"], "expected": 8.5 / 9}

    # Decision rules.
    results["decision"] = decide_primary(
        {
            "delta_auc": 0.04,
            "delta_auc_interval": [0.025, 0.055],
            "delta_brier": 0.01,
            "delta_brier_interval": [0.006, 0.014],
        },
        calibration_error_increase=(0.001, 0.004),
    ).to_row()

    # Intervention identity and matching on a small synthetic vector.
    w_down = torch.randn(8, 32)
    v_norms = w_down.norm(dim=0)
    s = torch.randn(32)
    ident = identity(s, w_down)
    top = attenuate_topk(s, w_down, v_norms, k=3)
    random = random_matched(
        s, w_down, v_norms, k=3, target_mass=top.removed_mass,
        target_norm=top.perturbation_norm, seed=1,
    )
    results["interventions"] = {
        "identity_error": float((ident.o_new - w_down @ s).abs().max()),
        "top_mass": top.removed_mass,
        "random_mass": random.removed_mass,
        "random_matched": random.matched,
    }

    results["passed"] = bool(results["preprocessing_train_only"]["passed"]) and abs(
        results["within_question"]["value"] - 8.5 / 9
    ) < 1e-9 and results["interventions"]["identity_error"] < 1e-5
    return results


# ---------------------------------------------------------------------------
# Level 3: tiny end-to-end collection
# ---------------------------------------------------------------------------
def smoke_end_to_end(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", limit: int = 2) -> dict:
    from .data.cohort import CohortQuestion
    from .models.collect import CollectSettings, collect
    from .models.loader import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(model_name, dtype="float16" if torch.cuda.is_available() else "float32")
    questions = [
        CohortQuestion(
            question_id=f"smoke-{index}",
            subject_id=f"subject-{index}",
            subject_name="Test Subject",
            relation="father",
            question="Who is the father of George W. Bush?",
            reference_answers=["George H. W. Bush"],
            split="dev",
        )
        for index in range(limit)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths = run_paths(Path(tmp))
        settings = CollectSettings(batch_questions=1, max_questions=limit, resume=False)
        outputs = {}
        outputs["track_a"] = collect(loaded, questions, settings, paths, track="A")
        outputs["track_b"] = collect(loaded, questions, settings, paths, track="B")
        checks = {}
        for name, output in outputs.items():
            for kind, path in output.items():
                text = Path(path).read_text()
                checks[f"{name}_{kind}_lines"] = len([line for line in text.splitlines() if line.strip()])
        feature_lines = []
        for path in paths["features"].glob("*.jsonl"):
            feature_lines.extend(line for line in path.read_text().splitlines() if line.strip())
        reference_leak = any(
            "George" in line or '"label"' in line for line in feature_lines
        )
        checks["feature_rows"] = len(feature_lines)
        checks["reference_or_label_in_features"] = reference_leak
        checks["passed"] = (
            checks.get("track_a_labels_lines", 0) >= limit
            and checks.get("track_b_labels_lines", 0) >= 1
            and checks.get("track_a_features_lines", 0) >= limit
            and not reference_leak
        )
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Section 3.3 smoke checks")
    parser.add_argument("--end-to-end", action="store_true", help="also run the small-model collection")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--json", type=Path, default=None, help="write the report to this file")
    args = parser.parse_args()

    report = {"contracts": smoke_contracts(), "pipeline": smoke_pipeline()}
    if args.end_to_end:
        report["end_to_end"] = smoke_end_to_end(args.model)
    report["all_passed"] = all(
        bool(section.get("passed")) for section in report.values() if isinstance(section, dict)
    )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    log("smoke checks passed" if report["all_passed"] else "SMOKE CHECKS FAILED")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
