"""Command line entry points for the retrieval-monitoring package.

    python -m retrieval_monitoring.cli text-baselines --run runs/section33_confirmatory
    python -m retrieval_monitoring.cli check-state --model Qwen/Qwen2.5-0.5B-Instruct
    python -m retrieval_monitoring.cli toy --steps 300
    python -m retrieval_monitoring.cli pilot --run runs/section33_confirmatory \
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_text_baselines(args) -> int:
    import pandas as pd

    from .text_baselines import run_track, save_reports

    reports, score_frames = [], []
    for track in args.tracks:
        modes = ("question",) if track.upper() == "A" else ("question", "question_candidate")
        track_reports, scores = run_track(args.run, track, modes)
        reports.extend(track_reports)
        score_frames.append(scores)
        for report in track_reports:
            within = report.within_question_auroc
            within_text = "n/a" if within is None else f"{within:.3f}"
            print(
                f"track {report.track} mode {report.mode}: "
                f"overall {report.overall_auroc:.3f}, "
                f"within {within_text} ({report.eligible_questions} mixed), "
                f"saved H {report.saved.get('saved_H', float('nan')):.3f}, "
                f"observed delta {report.observed_delta_vs_hidden:+.3f} "
                f"[{report.delta_ci[0]:+.3f}, {report.delta_ci[1]:+.3f}]"
            )
    path = save_reports(args.run, reports, pd.concat(score_frames, ignore_index=True))
    print(f"wrote {path}")
    return 0


def cmd_check_state(args) -> int:
    from section33.models.loader import load_model_and_tokenizer

    from .checks import run_state_check

    loaded = load_model_and_tokenizer(args.model, dtype=args.dtype)
    checks = run_state_check(loaded, layer_index=args.layer)
    print(json.dumps(checks, indent=2))
    return 0 if checks["passed"] else 1


def cmd_toy(args) -> int:
    from .toy_world import ToyConfig, run_toy

    config = ToyConfig(
        entities=args.entities,
        dim=args.dim,
        steps=args.steps,
        seeds=args.seeds,
    )
    results = run_toy(config)
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


def cmd_fetch_pairs(args) -> int:
    from .pairs import fetch_pairs, write_pairs_csv

    frame = fetch_pairs(tuple(args.relations), limit=args.limit, endpoint=args.endpoint)
    path = write_pairs_csv(frame, args.out)
    print(f"wrote {path}: {len(frame)} subjects with relations {args.relations}")
    return 0


def cmd_sweep(args) -> int:
    from section33.models.loader import load_model_and_tokenizer

    from .pairs import load_pairs_csv
    from .runner import load_normalized_source
    from .sweep import SweepConfig, run_sweep

    source = load_normalized_source(args.source)
    pairs = load_pairs_csv(args.pairs, tuple(args.relations))[: args.subjects]
    loaded = load_model_and_tokenizer(args.model, dtype=args.dtype)
    config = SweepConfig(
        layers=tuple(args.layers),
        ranks=tuple(args.ranks),
        alphas=tuple(args.alphas),
        n_subjects=args.subjects,
        relations=tuple(args.relations),
        map_subjects=args.map_subjects,
    )
    out_dir = args.out or (
        Path(args.run) / "retrieval_monitoring" if args.run else None
    )
    report = run_sweep(loaded, source, pairs, config, run_dir=args.run, out_dir=out_dir)
    print(json.dumps(report, indent=2))
    return 0


def cmd_pilot(args) -> int:
    from .runner import run_pilot, run_pilot_from_source

    relations = tuple(args.relations)
    if args.source is not None:
        report = run_pilot_from_source(
            args.source,
            args.model,
            run_dir=args.run,
            pairs_csv=args.pairs,
            n_subjects=args.subjects,
            relations=relations,
            layer_index=args.layer,
            alpha=args.alpha,
            rank=args.rank,
            dtype=args.dtype,
        )
    else:
        report = run_pilot(
            args.run,
            args.model,
            n_subjects=args.subjects,
            relations=relations,
            layer_index=args.layer,
            alpha=args.alpha,
            rank=args.rank,
            dtype=args.dtype,
        )
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_monitoring")
    sub = parser.add_subparsers(dest="command", required=True)

    text = sub.add_parser("text-baselines", help="S0 repaired input-only baselines")
    text.add_argument("--run", type=Path, default=Path("runs/section33_confirmatory"))
    text.add_argument("--tracks", nargs="+", default=["A", "B"])
    text.set_defaults(func=cmd_text_baselines)

    state = sub.add_parser("check-state", help="S1 retained-state apparatus checks")
    state.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    state.add_argument("--layer", type=int, default=None)
    state.add_argument("--dtype", default="float32")
    state.set_defaults(func=cmd_check_state)

    toy = sub.add_parser("toy", help="S2 one-layer toy world")
    toy.add_argument("--entities", type=int, default=512)
    toy.add_argument("--dim", type=int, default=64)
    toy.add_argument("--steps", type=int, default=500)
    toy.add_argument("--seeds", type=int, default=3)
    toy.add_argument("--out", default=None)
    toy.set_defaults(func=cmd_toy)

    fetch = sub.add_parser(
        "fetch-pairs", help="fetch subjects with two relations (login node, needs internet)"
    )
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--limit", type=int, default=200)
    fetch.add_argument("--relations", nargs=2, default=["father", "occupation"])
    fetch.add_argument("--endpoint", default="https://query.wikidata.org/sparql")
    fetch.set_defaults(func=cmd_fetch_pairs)

    sweep = sub.add_parser("sweep", help="S3 development sweep over layer, rank, alpha")
    sweep.add_argument("--source", type=Path, required=True)
    sweep.add_argument("--pairs", type=Path, required=True)
    sweep.add_argument("--run", type=Path, default=None)
    sweep.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    sweep.add_argument("--subjects", type=int, default=12)
    sweep.add_argument("--layers", nargs="+", type=int, default=[8, 13, 18])
    sweep.add_argument("--ranks", nargs="+", type=int, default=[2, 8])
    sweep.add_argument("--alphas", nargs="+", type=float, default=[1.0, 5.0, 10.0, 25.0])
    sweep.add_argument("--map-subjects", type=int, default=200)
    sweep.add_argument("--relations", nargs=2, default=["father", "occupation"])
    sweep.add_argument("--dtype", default="float16")
    sweep.add_argument("--out", type=Path, default=None)
    sweep.set_defaults(func=cmd_sweep)

    pilot = sub.add_parser("pilot", help="S3 selectivity pilot")
    pilot.add_argument("--run", type=Path, default=Path("runs/section33_confirmatory"))
    pilot.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    pilot.add_argument("--subjects", type=int, default=40)
    pilot.add_argument("--layer", type=int, default=None)
    pilot.add_argument("--rank", type=int, default=2)
    pilot.add_argument("--alpha", type=float, default=0.5)
    pilot.add_argument("--dtype", default="float16")
    pilot.add_argument(
        "--source", type=Path, default=None,
        help="raw source CSV used to fit the relation maps",
    )
    pilot.add_argument(
        "--pairs", type=Path, default=None,
        help="pairs CSV from fetch-pairs; subjects with both relations",
    )
    pilot.add_argument("--relations", nargs=2, default=["father", "occupation"])
    pilot.set_defaults(func=cmd_pilot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
