"""Command line entry points for the Section 3.3 experiment.

Stages follow plan section 13.  Example cluster sequence:

    python -m section33.cli setup
    python -m section33.cli cohort --source data/wikidata_cohort.csv \\
        --seen results/pilot7b_labels.csv --per-relation 1500
    python -m section33.cli collect --model Qwen/Qwen2.5-7B-Instruct --track A --split train
    python -m section33.cli collect --model Qwen/Qwen2.5-7B-Instruct --track B --split train
    python -m section33.cli develop --model Qwen/Qwen2.5-7B-Instruct --track A
    python -m section33.cli evaluate --model Qwen/Qwen2.5-7B-Instruct --track A
    python -m section33.cli audit

No command here downloads weights or starts a long run by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import __version__
from .analysis.evaluation import (
    bootstrap_p_value,
    calibration_guardrail_interval,
    decide_equivalence,
    decide_primary,
    decide_within_question,
    holm,
    paired_subject_bootstrap,
    paired_within_question_bootstrap,
)
from .analysis.power import (
    development_variance,
    paired_development_distribution,
    power_table,
    within_question_power_table,
)
from .config import PROTOCOL, prompt_manifest, run_paths, write_protocol
from .data.cohort import (
    CohortQuestion,
    build_cohort,
    freeze_splits,
    load_seen_subjects,
    manifest_hash,
    validate_cohort,
)
from .datasets import expand_by_multiplicity, load_split, metadata_features
from .models.collect import CollectSettings, collect
from .models.loader import load_model_and_tokenizer
from .probes.calibration import fit_calibrator
from .probes.pipeline import (
    assemble_blocks,
    fit_probe,
    missing_blocks,
    spec_is_usable,
    standard_specs,
)
from .utils import ensure_dirs, log, set_seed, sha256_file, sha256_text, write_json


def _slug(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").lower()


def _write_schema_readme(path: Path, title: str, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(f"# {title}\n\n{text}\n")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def cmd_setup(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    protocol_hash = write_protocol(paths["protocol"] / "protocol.json")
    write_json(paths["protocol"] / "prompt_manifest.json", prompt_manifest())
    _write_schema_readme(
        paths["protocol"], "protocol",
        "Frozen design: margins, splits, prompts, budgets. Do not edit after the "
        "final test is opened. `protocol.json` carries the content hash.",
    )
    _write_schema_readme(
        paths["manifests"], "manifests",
        "Subject lists, split assignments, source revisions, and hashes. One row "
        "per question: question_id, subject_id, relation, stratum, split.",
    )
    _write_schema_readme(
        paths["labels"], "labels",
        "Question references, candidate answers, correctness labels, and parsed "
        "confidence reports. This interface holds the reference answers.",
    )
    _write_schema_readme(
        paths["features"], "features",
        "Reference-free feature rows. No reference answer and no label may appear "
        "in these files. Token IDs and seeds are saved for replay.",
    )
    _write_schema_readme(
        paths["probes"], "probes",
        "Fitted preprocessing pipelines, readouts, and calibrators, serialised with "
        "their feature order and training manifest hash.",
    )
    _write_schema_readme(
        paths["predictions"], "predictions",
        "One row per model/subject/question/answer/method with raw and calibrated "
        "scores, fold, seed, and label-version hash.",
    )
    _write_schema_readme(
        paths["evaluation"], "evaluation",
        "Paired intervals, subgroup counts, reliability tables, and decision.json.",
    )
    _write_schema_readme(
        paths["interventions"], "interventions",
        "Intervention settings, matched controls, regenerated outcomes.",
    )
    _write_schema_readme(
        paths["report"], "report",
        "Claim decisions, tables, plots, and failure examples.",
    )
    log(f"run layout ready at {paths['root']} (protocol {protocol_hash[:12]})")
    return {"run": str(paths["root"]), "protocol_hash": protocol_hash}


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------
def cmd_cohort(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    seen = load_seen_subjects([Path(path) for path in args.seen]) if args.seen else set()
    df, _ = build_cohort(
        Path(args.source),
        seen,
        per_relation=args.per_relation,
        seed=args.seed,
        source_revision=args.revision,
        allow_seen=args.allow_seen,
    )
    if df.empty:
        raise SystemExit("cohort is empty; check the source and exclusion list")
    sizes = {
        "train": args.train,
        "dev": args.dev,
        "calibration": args.calibration,
        "test": args.test,
    }
    df, split_manifest = freeze_splits(df, sizes=sizes, seed=args.seed)
    validation = validate_cohort(
        df, sizes, mode=getattr(args, "mode", "pilot"), excluded_subjects=len(seen)
    )
    if validation["problems"]:
        for problem in validation["problems"]:
            log(f"cohort validation: {problem}")
    df.to_parquet(paths["manifests"] / "cohort.parquet", index=False)
    manifest = {
        "source": args.source,
        "source_hash": sha256_file(Path(args.source)) if Path(args.source).exists() else None,
        "excluded_subjects": len(seen),
        "sizes_requested": sizes,
        **split_manifest,
        "relation_counts": df["relation"].value_counts().to_dict(),
        "stratum_counts": df["stratum"].value_counts().to_dict(),
        "design_validation": validation,
    }
    write_json(paths["manifests"] / "cohort_manifest.json", manifest)
    log(f"cohort ready: {len(df)} questions, {df['subject_id'].nunique()} subjects")
    return manifest


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def cmd_collect(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    cohort_path = paths["manifests"] / "cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(
            f"cohort manifest not found: {cohort_path}.  "
            "Run cluster/04_cohort.sbatch before collecting."
        )
    cohort = pd.read_parquet(cohort_path)
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    if args.shards > 1 and args.shard is None:
        raise SystemExit(
            "--shards > 1 requires an explicit --shard.  Use a Slurm job array "
            "(--array=0-N) or pass --shard INDEX.  See cluster/05_collect.sbatch."
        )
    shard = 0 if args.shard is None else args.shard
    if shard < 0 or shard >= args.shards:
        log(f"shard {shard} is outside [0, {args.shards - 1}]; nothing to do")
        return {"skipped": True, "split": args.split, "shard": shard, "shards": args.shards}
    block = cohort[cohort["split"] == args.split]
    if args.shards > 1:
        shard_of = block["subject_id"].map(
            lambda subject: int(sha256_text(str(subject))[:12], 16) % args.shards
        )
        block = block[shard_of == shard]
    if args.limit:
        block = block.head(args.limit)
    if block.empty:
        raise SystemExit("no questions in this split and shard")
    tag = args.tag or (f"shard{shard}of{args.shards}" if args.shards > 1 else "")
    questions = [
        CohortQuestion(
            question_id=str(row.question_id),
            subject_id=str(row.subject_id),
            subject_name=str(row.subject_name),
            relation=str(row.relation),
            question=str(row.question),
            reference_answers=list(row.reference_answers),
            split=str(row.split),
            popularity=float(row.popularity) if pd.notna(row.popularity) else None,
            stratum=str(row.stratum),
            source_revision=str(row.source_revision),
        )
        for row in block.itertuples()
    ]
    loaded = load_model_and_tokenizer(args.model, revision=args.revision, dtype=args.dtype)
    settings = CollectSettings(
        batch_questions=args.batch,
        max_questions=args.limit,
        generation_seed=args.seed,
        resume=not args.no_resume,
        tag=tag,
    )
    outputs = collect(loaded, questions, settings, paths, track=args.track)
    log(f"collected track {args.track} for {len(questions)} questions")
    return outputs


# ---------------------------------------------------------------------------
# development: feature selection and power simulation
# ---------------------------------------------------------------------------
def _add_metadata_columns(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    meta = metadata_features(frame, track=track)
    for column in meta.columns:
        frame[f"f__meta__{column}"] = meta[column].to_numpy()
    if "conf" in frame.columns:
        # V comparator: the stated confidence, kept out of the D block.
        frame["f__conf__stated"] = frame["conf"].astype(float)
    return frame


def _expand_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Make every sampled answer count once before any fitting or scoring.

    Training, validation, calibration, development scoring, and evaluation all
    see the same eight-draw population.  The weight column is reset to one so a
    later expansion cannot double-count the rows.
    """

    frame = expand_by_multiplicity(frame)
    frame["multiplicity"] = 1
    return frame


def cmd_develop(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    slug = _slug(args.model)
    frame = load_split(
        paths, slug, args.track, splits=("train", "dev"),
        expected_samples=PROTOCOL.track_b_samples if args.track.upper() == "B" else None,
    )
    if frame.empty:
        raise SystemExit("no development data; run collect for train and dev first")
    frame = _add_metadata_columns(frame, args.track)
    frame = _expand_samples(frame)
    train = frame[frame["split"] == "train"]
    dev = frame[frame["split"] == "dev"]
    # Keep only specs whose blocks all have matching features for this track and
    # frame.  Empty blocks are rejected rather than fitted as constants.
    specs = {
        name: spec
        for name, spec in standard_specs().items()
        if spec_is_usable(frame, spec)
    }
    if not specs:
        raise SystemExit("no probe specification has matching features; check the collection")
    results = {}
    predictions = {}
    for name, spec in specs.items():
        probe, best = fit_probe(train, train["label"].to_numpy(), train["subject_id"].to_numpy(), spec, seed=args.seed)
        dev_predictions = probe.predict(assemble_blocks(dev, spec))
        predictions[name] = dev_predictions
        results[name] = {
            "inner": best,
            "dev_brier": float(brier_score_loss(dev["label"], np.clip(dev_predictions, 0, 1))),
            "dev_auroc": float(roc_auc_score(dev["label"], dev_predictions))
            if len(dev["label"].unique()) > 1 else None,
            "n_columns": int(sum(assemble_blocks(dev, spec)[block.name].shape[1] for block in spec.blocks)),
        }
    selection = {
        "model": args.model,
        "track": args.track,
        "results": results,
        "note": "selection uses the training split only; development is reporting only",
    }
    write_json(paths["probes"] / f"dev_selection__{slug}__track{args.track}.json", selection)

    baseline_key = "B_after" if args.track.upper() == "B" else "B"
    comparison_key = "B_plus_M_after" if args.track.upper() == "B" else "B_plus_M"
    if baseline_key in predictions and comparison_key in predictions:
        dev = dev.copy()
        dev["score_B"] = predictions[baseline_key]
        dev["score_B_plus_M"] = predictions[comparison_key]
        distribution = paired_development_distribution(
            dev, "score_B_plus_M", "score_B", seed=args.seed
        )
        variance = development_variance(distribution)
        if not (np.isfinite(variance["se_auc"]) and np.isfinite(variance["se_brier"])):
            log("development variance is not estimable; skipping the power table")
        else:
            correlation = variance["corr"] if np.isfinite(variance["corr"]) else 0.8
            power = power_table(
                variance["se_auc"], variance["se_brier"],
                n_development=variance["n_subjects"],
                correlation=correlation, seed=args.seed,
            )
            write_json(paths["protocol"] / f"power__{slug}__track{args.track}.json", power)
            log(
                f"development variance: se_auc={variance['se_auc']:.4f}, "
                f"se_brier={variance['se_brier']:.4f}, n={variance['n_subjects']}"
            )
            if args.track.upper() == "B" and np.isfinite(variance["se_within"]):
                within_power = within_question_power_table(
                    variance["se_within"],
                    n_development=variance["n_subjects"],
                    seed=args.seed,
                    confidence=1.0 - 0.05 / len(PROTOCOL.secondary_family),
                )
                write_json(
                    paths["protocol"] / f"power_within__{slug}__trackB.json", within_power
                )
    return selection


# ---------------------------------------------------------------------------
# evaluation: frozen comparison on the final test
# ---------------------------------------------------------------------------
def _strip_samples(result: dict) -> dict:
    """Drop the 10k bootstrap sample lists before writing JSON."""

    return {key: value for key, value in result.items() if not key.endswith("_samples")}


def _verify_protocol(paths) -> dict:
    """Require the frozen protocol and prompt manifest before evaluation."""

    protocol_path = paths["protocol"] / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit(f"frozen protocol missing: {protocol_path}; run setup first")
    stored = json.loads(protocol_path.read_text())
    if stored.get("protocol_hash") != PROTOCOL.content_hash():
        raise SystemExit(
            "protocol hash mismatch: the frozen protocol does not match the current "
            "code; evaluation refused"
        )
    prompt_path = paths["protocol"] / "prompt_manifest.json"
    if not prompt_path.exists():
        raise SystemExit(f"prompt manifest missing: {prompt_path}; run setup first")
    if json.loads(prompt_path.read_text()) != prompt_manifest():
        raise SystemExit("prompt manifest mismatch: prompt text changed after setup")
    return stored


def _fitted_artifact(probe, calibrator, spec, train) -> dict:
    """Serialisable fitted pipeline for provenance (review F12)."""

    artifact = {
        "spec": spec.name,
        "classifier": spec.classifier,
        "blocks": [block.name for block in spec.blocks],
        "training_subject_hash": manifest_hash(train["subject_id"].unique()),
        "n_training_rows": int(len(train)),
        "calibrator": {"method": calibrator.method},
    }
    classifier = probe.classifier
    if classifier is not None and hasattr(classifier, "coef_"):
        artifact["coefficients"] = np.asarray(classifier.coef_, dtype=float).tolist()
        artifact["intercept"] = np.asarray(classifier.intercept_, dtype=float).tolist()
    if calibrator.method == "platt" and calibrator.model is not None:
        artifact["calibrator"]["coef"] = np.asarray(calibrator.model.coef_, dtype=float).tolist()
        artifact["calibrator"]["intercept"] = np.asarray(calibrator.model.intercept_, dtype=float).tolist()
    if calibrator.method == "isotonic" and calibrator.model is not None:
        artifact["calibrator"]["x_thresholds"] = np.asarray(calibrator.model.X_thresholds_, dtype=float).tolist()
        artifact["calibrator"]["y_thresholds"] = np.asarray(calibrator.model.y_thresholds_, dtype=float).tolist()
    return artifact


def _difficulty_report(frame, test, method: str) -> dict | None:
    """Descriptive difficulty, reliability, and coverage checks (review F11)."""

    from sklearn.linear_model import LogisticRegression

    from .analysis.difficulty import residual_correlation
    from .probes.calibration import reliability_bins, risk_coverage

    meta_columns = [c for c in frame.columns if c.startswith("f__meta__")]
    train = frame[frame["split"] == "train"]
    if not meta_columns or train.empty or test.empty or test["label"].nunique() < 2:
        return None
    model = LogisticRegression(max_iter=3000, C=0.1)
    model.fit(train[meta_columns].to_numpy(), train["label"].to_numpy())
    difficulty = model.predict_proba(test[meta_columns].to_numpy())[:, 1]
    labels = test["label"].to_numpy()
    scores = test[f"cal_{method}"].to_numpy()
    high = difficulty >= np.median(difficulty)

    def auc(mask) -> float | None:
        if len(np.unique(labels[mask])) < 2:
            return None
        return float(roc_auc_score(labels[mask], scores[mask]))

    return {
        "difficulty_model_train_auroc": float(
            roc_auc_score(train["label"], model.predict_proba(train[meta_columns].to_numpy())[:, 1])
        )
        if train["label"].nunique() > 1
        else None,
        "method_auroc_overall": auc(np.ones(len(labels), dtype=bool)),
        "method_auroc_high_difficulty": auc(high),
        "method_auroc_low_difficulty": auc(~high),
        "residual_correlation": residual_correlation(labels - scores, difficulty),
        "reliability": reliability_bins(scores, labels),
        "risk_coverage": risk_coverage(scores, labels),
        "note": "difficulty uses pre-answer metadata only; median split is descriptive",
    }


def cmd_evaluate(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    protocol = _verify_protocol(paths)
    slug = _slug(args.model)
    track = args.track.upper()
    frame = load_split(
        paths, slug, track, splits=("train", "calibration", "test"),
        expected_samples=PROTOCOL.track_b_samples if track == "B" else None,
    )
    if frame.empty:
        raise SystemExit("no evaluation data; run collect first")
    frame = _add_metadata_columns(frame, track)
    frame = _expand_samples(frame)
    specs = standard_specs()

    base = "B_after" if track == "B" else "B"
    augment = "B_plus_M_after" if track == "B" else "B_plus_M"
    hidden = "H_512_after" if track == "B" else "H_512"
    hidden_plus = "H_all_plus_M_after" if track == "B" else "H_all_plus_M"
    mlp = "M_all_after" if track == "B" else "M_all"
    v_name = "V_after" if track == "B" else "V"
    p_name = "P_after" if track == "B" else "P"
    control_capacity = "H_M_256_256_after" if track == "B" else "H_M_256_256"
    control_hidden = "H_768_after" if track == "B" else "H_768"
    control_nonlinear = "H_nonlinear_after" if track == "B" else "H_nonlinear"
    method_names = list(dict.fromkeys((
        base, augment, hidden, hidden_plus, mlp, v_name, p_name,
        control_capacity, control_hidden, control_nonlinear,
    )))

    missing = {
        name: missing_blocks(frame, specs[name])
        for name in method_names
        if not spec_is_usable(frame, specs[name])
    }
    if missing:
        raise SystemExit(f"required probes have no matching features: {missing}")

    train = frame[frame["split"] == "train"]
    calibration = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "test"].copy()
    if train.empty or calibration.empty or test.empty:
        raise SystemExit("train, calibration, and test splits must all be non-empty")

    outputs = {}
    feature_manifest = {}
    fitted_artifacts = {}
    for name in method_names:
        spec = specs[name]
        blocks = assemble_blocks(train, spec)
        feature_manifest[name] = {
            "classifier": spec.classifier,
            "dimensions": {block: int(matrix.shape[1]) for block, matrix in blocks.items()},
            "total_columns": int(sum(matrix.shape[1] for matrix in blocks.values())),
        }
        probe, best = fit_probe(
            train, train["label"].to_numpy(), train["subject_id"].to_numpy(), spec, seed=args.seed
        )
        raw_calibration = probe.predict(assemble_blocks(calibration, spec))
        calibrator, calibration_info = fit_calibrator(
            raw_calibration, calibration["label"].to_numpy(), calibration["subject_id"].to_numpy()
        )
        fitted_artifacts[name] = _fitted_artifact(probe, calibrator, spec, train)
        test[f"cal_{name}"] = calibrator.apply(probe.predict(assemble_blocks(test, spec)))
        outputs[name] = {
            "inner": best,
            "calibration": calibration_info,
            "test_auroc": float(roc_auc_score(test["label"], test[f"cal_{name}"]))
            if test["label"].nunique() > 1
            else None,
        }
        log(f"{name}: test AUROC {outputs[name]['test_auroc']}")

    write_json(
        paths["probes"] / f"fitted__{slug}__track{track}.json",
        {"protocol_hash": protocol["protocol_hash"], "methods": fitted_artifacts},
    )

    # The frame was expanded on load, so per-method and paired metrics use the
    # same eight-draw population (plan section 5.2).
    metric_frame = test
    family_confidence = 1.0 - 0.05 / len(PROTOCOL.secondary_family)

    def paired(a: str, b: str, confidence: float = 0.95) -> dict:
        return paired_subject_bootstrap(
            metric_frame, f"cal_{a}", f"cal_{b}", "label",
            subject="subject_id", strata=("relation", "stratum"), seed=args.seed,
            confidence=confidence,
        )

    primary = paired(augment, base)
    guardrail = calibration_guardrail_interval(
        metric_frame, f"cal_{augment}", f"cal_{base}", "label",
        subject="subject_id", strata=("relation", "stratum"), seed=args.seed,
    )
    decision = decide_primary(primary, calibration_error_increase=guardrail)

    secondary_specs = {
        "replication_hidden_plus_mlp_minus_hidden": (hidden_plus, hidden),
        "mlp_minus_hidden": (mlp, hidden),
        "augmented_minus_confidence": (augment, v_name),
        "augmented_minus_probability": (augment, p_name),
    }
    secondary = {}
    raw_p_values = []
    for key, (a, b) in secondary_specs.items():
        result = paired(a, b, confidence=family_confidence)
        p_value = bootstrap_p_value(result["delta_auc_samples"])
        secondary[key] = {**_strip_samples(result), "comparison": a, "baseline": b, "p_auc": p_value}
        raw_p_values.append(p_value)
    for key, adjusted in zip(secondary, holm(raw_p_values)):
        secondary[key]["p_auc_holm"] = adjusted
    secondary["mlp_minus_hidden"]["decision"] = decide_equivalence(
        tuple(secondary["mlp_minus_hidden"]["delta_auc_interval"])
    ).to_row()

    within = None
    if track == "B":
        within = paired_within_question_bootstrap(
            metric_frame, f"cal_{augment}", f"cal_{base}", "label",
            question="question_id", subject="subject_id",
            strata=("relation", "stratum"), seed=args.seed,
            confidence=family_confidence,
        )
        brier_increase = (
            -primary["delta_brier_interval"][1],
            -primary["delta_brier_interval"][0],
        )
        within["brier_increase_interval"] = list(brier_increase)
        within["decision"] = decide_within_question(
            tuple(within["within_question_delta_interval"]), brier_increase
        ).to_row()

    payload = {
        "model": args.model,
        "track": track,
        "comparison": augment,
        "baseline": base,
        "protocol_hash": protocol["protocol_hash"],
        "label_policy_versions": sorted(
            test["label_policy_version"].dropna().unique().tolist()
        )
        if "label_policy_version" in test.columns
        else [],
        "feature_spec_versions": sorted(
            test["feature_spec_version"].dropna().unique().tolist()
        )
        if "feature_spec_version" in test.columns
        else [],
        "paired": _strip_samples(primary),
        "calibration_guardrail_increase": list(guardrail),
        "decision": decision.to_row(),
        "secondary": secondary,
        "holm_family": list(secondary_specs),
        "claim_family": list(PROTOCOL.secondary_family),
        "secondary_confidence": family_confidence,
        "within_question": within,
        "difficulty": _difficulty_report(frame, test, augment),
        "methods": outputs,
        "feature_manifest": feature_manifest,
    }
    write_json(paths["evaluation"] / f"decision__{slug}__track{track}.json", payload)
    test.to_parquet(
        paths["predictions"] / f"test_predictions__{slug}__track{track}.parquet", index=False
    )
    return payload


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def cmd_audit(args) -> dict:
    paths = run_paths(args.run)
    report = {"run": str(paths["root"]), "checks": {}, "passed": True}
    feature_files = list(paths["features"].glob("*.jsonl"))
    report["checks"]["feature_files"] = len(feature_files)
    for path in feature_files:
        text = path.read_text()
        if "reference_answers" in text or '"label"' in text:
            report["checks"][path.name] = "LEAK: reference or label in feature file"
            report["passed"] = False
    manifest_path = paths["manifests"] / "cohort_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        report["checks"]["cohort_manifest"] = {
            "n_questions": manifest.get("sizes_actual"),
            "relation_counts": manifest.get("relation_counts"),
            "design_validation": manifest.get("design_validation"),
        }
    else:
        report["checks"]["cohort_manifest"] = "missing"
        report["passed"] = False

    protocol_path = paths["protocol"] / "protocol.json"
    if protocol_path.exists():
        stored_hash = json.loads(protocol_path.read_text()).get("protocol_hash")
        report["checks"]["protocol_hash"] = stored_hash
        if stored_hash != PROTOCOL.content_hash():
            report["checks"]["protocol_mismatch"] = "stored protocol differs from the current code"
            report["passed"] = False
    else:
        report["checks"]["protocol_hash"] = "missing"
        report["passed"] = False

    prompt_path = paths["protocol"] / "prompt_manifest.json"
    if prompt_path.exists():
        if json.loads(prompt_path.read_text()) == prompt_manifest():
            report["checks"]["prompt_manifest"] = "ok"
        else:
            report["checks"]["prompt_manifest"] = "mismatch"
            report["passed"] = False
    else:
        report["checks"]["prompt_manifest"] = "missing"
        report["passed"] = False
    write_json(paths["evaluation"] / "audit.json", report)
    if not report["passed"]:
        raise SystemExit(f"audit failed; see {paths['evaluation'] / 'audit.json'}")
    return report


def cmd_smoke(args) -> dict:
    from .smoke import main as smoke_main

    raise SystemExit(smoke_main() or 0)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="section33", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="create the run layout and freeze the protocol")
    setup.add_argument("--run", type=Path, default=None)
    setup.set_defaults(func=cmd_setup)

    cohort = sub.add_parser("cohort", help="build a fresh cohort and split manifest")
    cohort.add_argument("--run", type=Path, default=None)
    cohort.add_argument("--source", required=True)
    cohort.add_argument("--seen", nargs="*", default=[], help="historical label files to exclude")
    cohort.add_argument("--per-relation", type=int, default=1500)
    cohort.add_argument("--train", type=int, default=PROTOCOL.splits.train)
    cohort.add_argument("--dev", type=int, default=PROTOCOL.splits.dev)
    cohort.add_argument("--calibration", type=int, default=PROTOCOL.splits.calibration)
    cohort.add_argument("--test", type=int, default=PROTOCOL.splits.test)
    cohort.add_argument("--seed", type=int, default=0)
    cohort.add_argument("--revision", default="")
    cohort.add_argument("--allow-seen", action="store_true")
    cohort.add_argument(
        "--mode", choices=["pilot", "confirmatory"], default="pilot",
        help="confirmatory rejects missing relations, unmet quotas, and empty exclusions",
    )
    cohort.set_defaults(func=cmd_cohort)

    collect_cmd = sub.add_parser("collect", help="collect one track and split")
    collect_cmd.add_argument("--run", type=Path, default=None)
    collect_cmd.add_argument("--model", required=True)
    collect_cmd.add_argument("--revision", default=None)
    collect_cmd.add_argument("--track", choices=["A", "B"], required=True)
    collect_cmd.add_argument("--split", required=True)
    collect_cmd.add_argument("--limit", type=int, default=0)
    collect_cmd.add_argument("--batch", type=int, default=4)
    collect_cmd.add_argument("--dtype", default="float16")
    collect_cmd.add_argument("--seed", type=int, default=0)
    collect_cmd.add_argument("--shards", type=int, default=1, help="split the split into N shards")
    collect_cmd.add_argument("--shard", type=int, default=None, help="shard index, 0-based")
    collect_cmd.add_argument("--tag", default="", help="tag appended to output file names")
    collect_cmd.add_argument("--no-resume", action="store_true")
    collect_cmd.set_defaults(func=cmd_collect)

    develop = sub.add_parser("develop", help="select probes on train and size the test")
    develop.add_argument("--run", type=Path, default=None)
    develop.add_argument("--model", required=True)
    develop.add_argument("--track", choices=["A", "B"], default="A")
    develop.add_argument("--seed", type=int, default=0)
    develop.set_defaults(func=cmd_develop)

    evaluate = sub.add_parser("evaluate", help="frozen evaluation and decision file")
    evaluate.add_argument("--run", type=Path, default=None)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--track", choices=["A", "B"], default="A")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(func=cmd_evaluate)

    audit = sub.add_parser("audit", help="check schemas and reference exclusion")
    audit.add_argument("--run", type=Path, default=None)
    audit.set_defaults(func=cmd_audit)

    smoke = sub.add_parser("smoke", help="run the smoke checks")
    smoke.add_argument("--end-to-end", action="store_true")
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_seed(0)
    result = args.func(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
