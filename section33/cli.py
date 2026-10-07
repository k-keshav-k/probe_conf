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
    calibration_guardrail_interval,
    decide_primary,
    paired_subject_bootstrap,
)
from .analysis.difficulty import within_question_auroc
from .analysis.power import development_variance, power_table
from .config import PROTOCOL, run_paths, write_protocol
from .data.cohort import (
    CohortQuestion,
    build_cohort,
    freeze_splits,
    load_seen_subjects,
)
from .datasets import load_split, metadata_features
from .models.collect import CollectSettings, collect
from .models.loader import load_model_and_tokenizer
from .probes.calibration import fit_calibrator
from .probes.pipeline import assemble_blocks, fit_probe, standard_specs
from .utils import ensure_dirs, log, set_seed, sha256_file, write_json


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
    df.to_parquet(paths["manifests"] / "cohort.parquet", index=False)
    manifest = {
        "source": args.source,
        "source_hash": sha256_file(Path(args.source)) if Path(args.source).exists() else None,
        "excluded_subjects": len(seen),
        "sizes_requested": sizes,
        **split_manifest,
        "relation_counts": df["relation"].value_counts().to_dict(),
        "stratum_counts": df["stratum"].value_counts().to_dict(),
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
    cohort = pd.read_parquet(paths["manifests"] / "cohort.parquet")
    block = cohort[cohort["split"] == args.split]
    if args.limit:
        block = block.head(args.limit)
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
    )
    outputs = collect(loaded, questions, settings, paths, track=args.track)
    log(f"collected track {args.track} for {len(questions)} questions")
    return outputs


# ---------------------------------------------------------------------------
# development: feature selection and power simulation
# ---------------------------------------------------------------------------
def _add_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    meta = metadata_features(frame)
    for column in meta.columns:
        frame[f"f__meta__{column}"] = meta[column].to_numpy()
    return frame


def _subject_deltas(dev: pd.DataFrame, score_a: str, score_b: str, label: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Per-subject paired differences of AUROC contributions and Brier scores."""

    auc_deltas, brier_deltas = [], []
    for subject, block in dev.groupby("subject_id"):
        y = block[label].to_numpy()
        a = block[score_a].to_numpy()
        b = block[score_b].to_numpy()
        if len(np.unique(y)) > 1:
            auc_deltas.append(roc_auc_score(y, a) - roc_auc_score(y, b))
        brier_deltas.append(brier_score_loss(y, np.clip(b, 0, 1)) - brier_score_loss(y, np.clip(a, 0, 1)))
    return np.asarray(auc_deltas), np.asarray(brier_deltas)


def cmd_develop(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    slug = _slug(args.model)
    frame = load_split(paths, slug, args.track, splits=("train", "dev"))
    if frame.empty:
        raise SystemExit("no development data; run collect for train and dev first")
    frame = _add_metadata_columns(frame)
    train = frame[frame["split"] == "train"]
    dev = frame[frame["split"] == "dev"]
    specs = standard_specs()
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
            "n_columns": int(sum(len(assemble_blocks(dev, spec)[block.name][0]) for block in spec.blocks)),
        }
    selection = {
        "model": args.model,
        "track": args.track,
        "results": results,
        "note": "selection uses the training split only; development is reporting only",
    }
    write_json(paths["probes"] / f"dev_selection__{slug}__track{args.track}.json", selection)

    if "B" in predictions and "B_plus_M" in predictions:
        dev = dev.copy()
        dev["score_B"] = predictions["B"]
        dev["score_B_plus_M"] = predictions["B_plus_M"]
        auc_deltas, brier_deltas = _subject_deltas(dev, "score_B_plus_M", "score_B")
        variance = development_variance(auc_deltas, brier_deltas, seed=args.seed)
        power = power_table(variance["sd_auc"], variance["sd_brier"], correlation=variance["corr"])
        write_json(paths["protocol"] / f"power__{slug}__track{args.track}.json", power)
        log(f"development variance: sd_auc={variance['sd_auc']:.4f}, sd_brier={variance['sd_brier']:.4f}")
    return selection


# ---------------------------------------------------------------------------
# evaluation: frozen comparison on the final test
# ---------------------------------------------------------------------------
def cmd_evaluate(args) -> dict:
    paths = run_paths(args.run)
    ensure_dirs(paths)
    slug = _slug(args.model)
    frame = load_split(paths, slug, args.track, splits=("train", "calibration", "test"))
    if frame.empty:
        raise SystemExit("no evaluation data; run collect first")
    frame = _add_metadata_columns(frame)
    specs = standard_specs()
    comparison = "B_plus_M_after" if args.track.upper() == "B" else "B_plus_M"
    baseline = "B_after" if args.track.upper() == "B" else "B"

    train = frame[frame["split"] == "train"]
    calibration = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "test"].copy()
    outputs = {}
    for name in (baseline, comparison):
        spec = specs[name]
        probe, best = fit_probe(train, train["label"].to_numpy(), train["subject_id"].to_numpy(), spec, seed=args.seed)
        raw_calibration = probe.predict(assemble_blocks(calibration, spec))
        calibrator, calibration_info = fit_calibrator(
            raw_calibration, calibration["label"].to_numpy(), calibration["subject_id"].to_numpy()
        )
        raw_test = probe.predict(assemble_blocks(test, spec))
        test[f"raw_{name}"] = raw_test
        test[f"cal_{name}"] = calibrator.apply(raw_test)
        outputs[name] = {"inner": best, "calibration": calibration_info}
        log(f"{name}: test AUROC {roc_auc_score(test['label'], test[f'cal_{name}']):.4f}")

    result = paired_subject_bootstrap(
        test, f"cal_{comparison}", f"cal_{baseline}", "label",
        subject="subject_id", strata=("relation", "stratum"), seed=args.seed,
    )
    guardrail = calibration_guardrail_interval(
        test, f"cal_{comparison}", f"cal_{baseline}", "label",
        subject="subject_id", strata=("relation", "stratum"), seed=args.seed,
    )
    decision = decide_primary(result, calibration_error_increase=guardrail)
    payload = {
        "model": args.model,
        "track": args.track,
        "comparison": comparison,
        "baseline": baseline,
        "paired": result,
        "calibration_guardrail_increase": list(guardrail),
        "decision": decision.to_row(),
        "methods": outputs,
    }
    if args.track.upper() == "B":
        within = within_question_auroc(
            test[f"cal_{comparison}"].to_numpy(), test["label"].to_numpy(),
            test["question_id"].to_numpy(), weights=test.get("multiplicity", pd.Series(1, index=test.index)).to_numpy(),
        )
        payload["within_question"] = within
    write_json(paths["evaluation"] / f"decision__{slug}__track{args.track}.json", payload)
    test.to_parquet(paths["predictions"] / f"test_predictions__{slug}__track{args.track}.parquet", index=False)
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
        }
    report["checks"]["protocol_hash"] = (
        json.loads((paths["protocol"] / "protocol.json").read_text()).get("protocol_hash")
        if (paths["protocol"] / "protocol.json").exists()
        else "missing"
    )
    if report["checks"]["protocol_hash"] == "missing":
        report["passed"] = False
    write_json(paths["evaluation"] / "audit.json", report)
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
