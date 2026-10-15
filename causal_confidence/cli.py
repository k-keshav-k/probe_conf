"""Command line entry points for the causal-confidence stages.

The experiment is staged (plan section 10): P0 (reporter positive control),
P1 (episode/answer-policy validation), P2 (intervention discovery + one locked
validation).  P3-P6 are gated and refuse to run before P2 passes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from . import data as data_module
from . import stages
from .p0 import collect_toy_rows, run_p0_controls, synthetic_rows
from .protocol import (
    PROTOCOL,
    Protocol,
    ensure_run_dirs,
    run_paths,
    write_protocol,
)

# The pooled cohort and PopQA source live in the section 33 run root.  Honour
# ``RUN_DIR`` (set by ``cluster/env.sh``) exactly like section33/config.py so
# cluster jobs read the synced data instead of a repo-local copy.
RUN_BASE = Path(os.environ.get("RUN_DIR", "runs/section33_v1"))
DEFAULT_COHORT = RUN_BASE / "manifests" / "cohort.parquet"
DEFAULT_POPQA = RUN_BASE / "source" / "popqa_source.parquet"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def smoke_protocol(protocol: Protocol, *, questions: int = 24) -> Protocol:
    """A reduced protocol for laptop/small-model smoke runs.

    Sizes shrink; the gates do not.  Smoke output is marked as smoke in the
    manifest and must never be reported as a pilot result.
    """

    return dataclasses.replace(
        protocol,
        partitions=dataclasses.replace(
            protocol.partitions,
            discovery=questions,
            validation=questions,
            train=64,
            dev=16,
            test=32,
            transfer=16,
        ),
        sampling=dataclasses.replace(
            protocol.sampling, train_draws=2, eval_draws=2
        ),
        intervention=dataclasses.replace(
            protocol.intervention,
            site_fractions=(1 / 3, 2 / 3),
            group_sizes=(8, 16),
            screening_questions=min(12, questions),
            screening_draws=2,
            refinement_attenuations=(0.25,),
            random_replicates=1,
        ),
    )


def build_pool(
    cohort_path: Path, popqa_path: Path, *, allow_reduced: bool
) -> tuple[dict, list]:
    rows: list = []
    sources = []
    if cohort_path.exists():
        rows.extend(data_module.load_cohort(cohort_path))
        sources.append(str(cohort_path))
    if popqa_path.exists():
        rows.extend(data_module.load_popqa(popqa_path))
        sources.append(str(popqa_path))
    if not rows:
        raise SystemExit(
            "no question sources found; pass --cohort/--popqa or build a cohort first"
        )
    rows, dedupe = data_module.dedupe_questions(rows)
    return {"sources": sources, "dedupe": dedupe, "allow_reduced": allow_reduced}, rows


def question_pool_for(
    protocol: Protocol,
    *,
    cohort_path: Path = DEFAULT_COHORT,
    popqa_path: Path = DEFAULT_POPQA,
    allow_reduced: bool = False,
):
    meta, rows = build_pool(cohort_path, popqa_path, allow_reduced=allow_reduced)
    partitions, allocation = data_module.build_partitions(
        rows,
        protocol.partitions.as_dict(),
        seed=protocol.inner_seed,
        allow_reduced=allow_reduced,
    )
    meta["allocation"] = allocation
    return partitions, meta


def write_split_manifest(paths: dict, partitions: dict, meta: dict, protocol: Protocol) -> dict:
    return data_module.write_manifest(
        partitions,
        paths["manifests"] / "cohort_split.json",
        allocation=meta,
        sources=meta["sources"],
        protocol=protocol,
    )


def _load_environment(args, *, name: str | None = None, smoke_model: bool = False):
    model = (
        name
        or getattr(args, "model", None)
        or (SMOKE_MODEL if getattr(args, "smoke", False) and smoke_model else PROTOCOL.model_name)
    )
    revision = None if model != PROTOCOL.model_name else PROTOCOL.model_revision
    return stages.load_model_and_tokenizer(
        model,
        revision=revision,
        device=getattr(args, "device", None),
        dtype=getattr(args, "dtype", "auto"),
    )


def _write_json(path: Path, payload: dict) -> str:
    from .outcomes import write_immutable

    return write_immutable(path, payload)


def command_setup(args) -> int:
    paths = ensure_run_dirs(args.run_root)
    digest = write_protocol(paths["protocol"] / "protocol.json", PROTOCOL)
    print(f"protocol written to {paths['protocol'] / 'protocol.json'} (hash {digest})")
    return 0


def command_p0(args) -> int:
    paths = ensure_run_dirs(args.run_root)
    if args.source == "toy":
        rows = collect_toy_rows(
            arm=args.arm, seed=PROTOCOL.inner_seed, batches=args.batches, smoke=args.smoke
        )
    elif args.source == "synthetic":
        rows = synthetic_rows(seed=PROTOCOL.inner_seed)
    else:
        raise SystemExit(f"unknown P0 source: {args.source}")
    result = run_p0_controls(
        rows,
        seed=PROTOCOL.inner_seed,
        head_steps=300 if args.smoke else 1_500,
    )
    out = Path(args.out) if args.out else paths["evaluation"] / "p0_reporter_controls.json"
    _write_json(out, result)
    print(json.dumps({key: result[key] for key in ("passed", "gates", "failure_analysis")}, indent=2))
    print(f"written: {out}")
    return 0 if result["passed"] else 1


def command_p1(args) -> int:
    paths = ensure_run_dirs(args.run_root)
    protocol = smoke_protocol(PROTOCOL) if args.smoke else PROTOCOL
    partitions, meta = question_pool_for(
        protocol,
        cohort_path=Path(args.cohort),
        popqa_path=Path(args.popqa),
        allow_reduced=args.allow_reduced or args.smoke,
    )
    manifest = write_split_manifest(paths, partitions, meta, protocol)
    env = _load_environment(args, smoke_model=True)
    discovery = partitions["discovery"]
    result, banks = stages.run_p1(env, discovery, draws=args.draws)
    result["manifest_hash"] = manifest["manifest_hash"]
    result["smoke"] = bool(args.smoke)
    _write_json(paths["evaluation"] / "p1_episode_contract.json", result)
    from .outcomes import write_banks

    write_banks(paths["outcomes"] / "p1_discovery_banks.json", banks, stage="p1")
    print(
        json.dumps(
            {
                "contract_passed": result["contract"]["passed"],
                "mean_success": result["answer_policy"]["mean_success"],
                "n_questions": result["n_questions"],
            },
            indent=2,
        )
    )
    return 0 if result["contract"]["passed"] else 1


def command_p2_discover(args) -> int:
    paths = ensure_run_dirs(args.run_root)
    protocol = smoke_protocol(PROTOCOL) if args.smoke else PROTOCOL
    partitions, meta = question_pool_for(
        protocol,
        cohort_path=Path(args.cohort),
        popqa_path=Path(args.popqa),
        allow_reduced=args.allow_reduced or args.smoke,
    )
    write_split_manifest(paths, partitions, meta, protocol)
    env = _load_environment(args, smoke_model=True)
    result = stages.run_p2_discovery(
        env, partitions["discovery"], protocol=protocol, draws=args.draws
    )
    result["smoke"] = bool(args.smoke)
    _write_json(paths["interventions"] / "p2_discovery.json", result)
    frozen = result["frozen"]
    print(
        json.dumps(
            {
                "target": frozen["target"],
                "mild": frozen["mild_attenuation"],
                "strong": frozen["strong_attenuation"],
                "rescue_recovery_fraction": frozen["rescue_recovery_fraction"],
                "candidates": [
                    {"label": item["group"]["label"], "damage": item["screening"]["mean_damage"]}
                    for item in result["candidates"]
                ],
            },
            indent=2,
        )
    )
    return 0


def command_p2_validate(args) -> int:
    paths = ensure_run_dirs(args.run_root)
    protocol = smoke_protocol(PROTOCOL) if args.smoke else PROTOCOL
    discovery_path = paths["interventions"] / "p2_discovery.json"
    if not discovery_path.exists():
        raise SystemExit("run p2-discover first")
    frozen_payload = json.loads(discovery_path.read_text())
    frozen = stages.FrozenIntervention.from_json(frozen_payload["frozen"])
    partitions, meta = question_pool_for(
        protocol,
        cohort_path=Path(args.cohort),
        popqa_path=Path(args.popqa),
        allow_reduced=args.allow_reduced or args.smoke,
    )
    write_split_manifest(paths, partitions, meta, protocol)
    env = _load_environment(args, smoke_model=True)
    result, banks = stages.run_p2_validation(
        env,
        partitions["validation"],
        frozen=frozen,
        protocol=protocol,
        draws=args.draws,
        random_replicates=args.random_replicates,
    )
    result["smoke"] = bool(args.smoke)
    from .outcomes import write_banks

    _write_json(paths["interventions"] / "p2_validation.json", result)
    write_banks(
        paths["outcomes"] / "p2_validation_banks.json",
        [bank for slots in banks.values() for bank in slots.values()],
        stage="p2_validation",
    )
    stages.plot_p2_validation(result, paths["figures"] / "p2_damage_restoration.png")
    stages.plot_p2_per_question(banks, paths["figures"] / "p2_per_question.png")
    print(json.dumps(result["gates"], indent=2))
    print(f"passed: {result['passed']}")
    return 0 if result["passed"] else 1


def command_gated(name: str):
    def run(args) -> int:
        try:
            getattr(stages, f"run_{name}")(args)
        except stages.StageGated as error:
            print(f"{name}: {error}")
            return 2
        return 0

    return run


def command_audit(args) -> int:
    """Re-hash frozen artifacts and check the run's manifest is present."""

    paths = run_paths(args.run_root)
    report: dict = {"root": str(paths["root"]), "files": {}}
    from .outcomes import check_bank_immutability

    for name in (
        "outcomes/p1_discovery_banks.json",
        "outcomes/p2_validation_banks.json",
    ):
        path = paths["root"] / name
        if path.exists():
            report["files"][name] = check_bank_immutability(path)
    protocol_path = paths["protocol"] / "protocol.json"
    if protocol_path.exists():
        report["protocol"] = json.loads(protocol_path.read_text()).get("protocol_hash")
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal_confidence", description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="write the frozen protocol")
    setup.set_defaults(func=command_setup)

    p0 = sub.add_parser("p0", help="reporter positive control")
    p0.add_argument("--source", choices=("toy", "synthetic"), default="toy")
    p0.add_argument("--arm", default="B", help="toy retrieval arm for data regeneration")
    p0.add_argument("--batches", type=int, default=40)
    p0.add_argument("--smoke", action="store_true")
    p0.add_argument("--out", default=None)
    p0.set_defaults(func=command_p0)

    def add_model_args(target):
        target.add_argument("--model", default=None)
        target.add_argument("--device", default=None)
        target.add_argument("--dtype", default="auto")
        target.add_argument("--cohort", default=str(DEFAULT_COHORT))
        target.add_argument("--popqa", default=str(DEFAULT_POPQA))
        target.add_argument("--allow-reduced", action="store_true")
        target.add_argument("--smoke", action="store_true")

    p1 = sub.add_parser("p1", help="episode contract and answer-policy audit")
    add_model_args(p1)
    p1.add_argument("--draws", type=int, default=8)
    p1.set_defaults(func=command_p1)

    p2d = sub.add_parser("p2-discover", help="bounded discovery search")
    add_model_args(p2d)
    p2d.add_argument("--draws", type=int, default=None)
    p2d.set_defaults(func=command_p2_discover)

    p2v = sub.add_parser("p2-validate", help="locked intervention validation")
    add_model_args(p2v)
    p2v.add_argument("--draws", type=int, default=None)
    p2v.add_argument("--random-replicates", type=int, default=None)
    p2v.set_defaults(func=command_p2_validate)

    for name in ("p3", "p4", "p5", "p6"):
        stage = sub.add_parser(name, help="gated stage (see plan section 10)")
        stage.set_defaults(func=command_gated(name))

    audit = sub.add_parser("audit", help="verify frozen artifacts")
    audit.set_defaults(func=command_audit)

    smoke = sub.add_parser("smoke", help="small-model end-to-end smoke")
    smoke.add_argument("--end-to-end", action="store_true")
    smoke.add_argument("--device", default=None)
    smoke.set_defaults(func=lambda args: _smoke(args))
    return parser


def _smoke(args) -> int:
    from .smoke import run_smoke

    return run_smoke(end_to_end=args.end_to_end, device=args.device)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
