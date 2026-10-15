"""Command line entry points for the selective-metacognition toy.

Examples::

    python -m selective_metacognition.cli e1 --seeds 3 --steps 400
    python -m selective_metacognition.cli e2 --seeds 3 --out runs/selective_metacognition/e2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ToyConfig


def _build_config(args) -> ToyConfig:
    overrides = {}
    for name in (
        "steps",
        "warm_start_steps",
        "batch",
        "lr",
        "warm_start_lr",
        "report_steps",
        "report_collect_steps",
        "rounds",
        "lambda_sel",
        "eta_consistency",
        "beta_kl",
        "spurious",
        "overlap",
        "d_model",
        "n_layers",
        "code_dim",
        "n_relations",
        "memory_dim",
        "adapter_rank",
        "d_report",
        "report_layers",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if getattr(args, "dense", False):
        overrides["dense"] = True
    if getattr(args, "seeds", None) is not None:
        overrides["seeds"] = args.seeds
    return ToyConfig(**overrides)


def _print_e1(report: dict) -> None:
    header = (
        f"{'arm':>10} {'split':>6} {'greedy':>7} {'sampled':>8} {'irr_stab':>8} "
        f"{'tgt_loss':>8} {'irr_loss':>8} {'select':>7} {'swap':>7} {'top1_share':>10}"
    )
    print(header)
    print("-" * len(header))
    for arm in ("warm_start", "B", "C"):
        for split in ("train", "test", "fresh"):
            metrics = report["aggregate"][arm][split]
            clean = metrics["clean"]["success_prob"]
            target_loss = clean - metrics["target_weaken"]["success_prob"]
            irr_loss = clean - metrics["irrelevant_weaken"]["success_prob"]
            print(
                f"{arm:>10} {split:>6} {metrics['clean']['top1']:>7.3f} {clean:>8.3f} "
                f"{metrics['irrelevant_swap_stability']:>8.3f} {target_loss:>8.3f} "
                f"{irr_loss:>8.3f} {metrics['selectivity']:>7.3f} "
                f"{metrics['target_swap_valid']['top1']:>7.3f} {metrics['top1_share']:>10.3f}"
            )


def _print_e2(report: dict) -> None:
    header = (
        f"{'arm':>4} {'greedy':>7} {'sampled':>8} {'report':>8} {'const':>7} {'recal':>7} "
        f"{'ordinary':>10} {'privileged':>11} {'sq_prob':>8} {'sign':>6}"
    )
    print(header)
    print("-" * len(header))
    for arm in ("A", "B", "C", "D", "E"):
        metrics = report["aggregate"][arm]
        functional = metrics["functional"]["test"]
        reporting = metrics["reporting_test"]
        print(
            f"{arm:>4} {functional['clean']['top1']:>7.3f} {functional['clean']['success_prob']:>8.3f} "
            f"{reporting['brier_report']:>8.4f} {reporting['brier_constant_half']:>7.4f} "
            f"{reporting['brier_report_recalibrated']:>7.4f} "
            f"{reporting['incremental_brier']:>+10.5f} {reporting['incremental_privileged']:>+11.5f} "
            f"{reporting['squared_probability_error']:>8.4f} "
            f"{reporting['same_question']['sign_agreement']:>6.3f}"
        )


def _save(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


def cmd_e1(args) -> int:
    from .runner import run_e1

    report = run_e1(_build_config(args), seeds=args.seeds)
    _print_e1(report)
    if args.out:
        _save(report, args.out)
    return 0


def cmd_e2(args) -> int:
    from .runner import run_e2

    report = run_e2(_build_config(args), seeds=args.seeds)
    _print_e2(report)
    if args.out:
        _save(report, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selective_metacognition")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, function in (("e1", cmd_e1), ("e2", cmd_e2)):
        command = sub.add_parser(name, help=f"run toy experiment {name.upper()}")
        command.add_argument("--seeds", type=int, default=3)
        command.add_argument("--steps", type=int, default=None)
        command.add_argument("--warm-start-steps", type=int, default=None)
        command.add_argument("--batch", type=int, default=None)
        command.add_argument("--lr", type=float, default=None)
        command.add_argument("--warm-start-lr", type=float, default=None)
        command.add_argument("--report-steps", type=int, default=None)
        command.add_argument("--report-collect-steps", type=int, default=None)
        command.add_argument("--rounds", type=int, default=None)
        command.add_argument("--lambda-sel", type=float, default=None)
        command.add_argument("--eta-consistency", type=float, default=None)
        command.add_argument("--beta-kl", type=float, default=None)
        command.add_argument("--spurious", type=float, default=None)
        command.add_argument("--overlap", type=float, default=None)
        command.add_argument("--dense", action="store_true")
        command.add_argument("--d-model", type=int, default=None)
        command.add_argument("--n-layers", type=int, default=None)
        command.add_argument("--code-dim", type=int, default=None)
        command.add_argument("--n-relations", type=int, default=None)
        command.add_argument("--memory-dim", type=int, default=None)
        command.add_argument("--adapter-rank", type=int, default=None)
        command.add_argument("--d-report", type=int, default=None)
        command.add_argument("--report-layers", type=int, default=None)
        command.add_argument("--out", type=Path, default=None)
        command.set_defaults(func=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
