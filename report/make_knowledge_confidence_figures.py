"""Figures for report/CAUSAL_CONFIDENCE_REPORT.md.

Reads the frozen artifacts written by the toy runs (E1-E3), the
causal-confidence stages (P0-P2), the knowledge-variant study, and the
readout probes.  Run from the repository root:

    .venv/bin/python report/make_knowledge_confidence_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures" / "knowledge_confidence"
OUT.mkdir(parents=True, exist_ok=True)

TOY = ROOT / "runs" / "section33_v1" / "selective_metacognition"
CC = ROOT / "runs" / "section33_v1" / "causal_confidence" / "cc_v1"
LOCAL_KNOW = ROOT / "runs" / "causal_confidence" / "cc_v1" / "knowledge"
DIAG = LOCAL_KNOW / "diagnostics"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_toy_rl_arms() -> None:
    e2 = load(TOY / "e2.json")["aggregate"]
    e3 = load(TOY / "e3.json")["aggregate"]
    arms = ["A", "B", "C", "D", "E"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(arms))
    report = [e2[arm]["reporting_test"]["brier_report"] for arm in arms]
    baseline = [e2[arm]["reporting_test"]["brier_ordinary_baseline"] for arm in arms]
    axes[0].bar(x - 0.2, report, width=0.4, label="trained report", color="tab:blue")
    axes[0].bar(x + 0.2, baseline, width=0.4, label="output-probability baseline", color="tab:gray")
    axes[0].axhline(0.25, color="black", linestyle="--", linewidth=0.8)
    axes[0].text(len(arms) - 0.5, 0.252, "always 50%", ha="right", fontsize=8)
    axes[0].set_xticks(x, arms)
    axes[0].set_ylim(0, 0.32)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].set_title("Toy E2: report vs baseline (test)")
    axes[0].legend(fontsize=8)

    arms3 = ["A", "B", "C", "E"]
    x3 = np.arange(len(arms3))
    test_added = [e3[arm]["prospective_test"]["incremental_brier"] for arm in arms3]
    held_added = [e3[arm]["prospective_heldout"]["incremental_brier"] for arm in arms3]
    axes[1].bar(x3 - 0.2, test_added, width=0.4, label="seen strengths", color="tab:blue")
    axes[1].bar(x3 + 0.2, held_added, width=0.4, label="unseen strengths", color="tab:orange")
    axes[1].set_xticks(x3, arms3)
    axes[1].set_ylabel("added Brier value (higher is better)")
    axes[1].set_title("Toy E3: added value of the report")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "fig1_toy_rl_arms.png", dpi=150)
    plt.close(figure)


def fig_p2_validation() -> None:
    discovery = load(CC / "p2c" / "interventions" / "p2_discovery.json")
    validation = load(CC / "p2c" / "interventions" / "p2_validation.json")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = []
    values = []
    for candidate in discovery["candidates"]:
        labels.append(candidate["group"]["label"])
        values.append(candidate["screening"]["mean_damage"])
    for refinement in discovery.get("refinements", []):
        labels.append(f'site14 a={refinement["attenuation"]}')
        values.append(refinement["mean_damage"])
    axes[0].bar(range(len(values)), values, color="tab:blue")
    axes[0].axhline(0.05, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].text(0.1, 0.051, "gate: 0.05", color="tab:red", fontsize=8)
    axes[0].set_xticks(range(len(values)), labels, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("success drop when damaged")
    axes[0].set_title("P2 discovery: candidate damage (100 questions)")

    effects = validation["effects"]
    names = ["mild", "strong", "matched\nrandom", "other\ngroup", "restoration\nrecovered"]
    values = [
        effects["damage_mild"],
        effects["damage_strong"],
        effects["random_damage"],
        effects["other_group_damage"],
        effects["restoration"]["recovered"] if effects["restoration"] else 0.0,
    ]
    errors = [
        0.0,
        effects["damage_strong_ci"]["ci_high"] - effects["damage_strong"],
        0.0,
        0.0,
        0.0,
    ]
    axes[1].bar(names, values, yerr=errors, capsize=4, color="tab:blue")
    axes[1].axhline(0.05, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.03, color="tab:orange", linestyle="--", linewidth=1.0)
    axes[1].text(0.0, 0.052, "damage gate 0.05", color="tab:red", fontsize=8)
    axes[1].text(0.0, 0.033, "target-random gate 0.03", color="tab:orange", fontsize=8)
    axes[1].set_ylim(-0.01, 0.06)
    axes[1].set_ylabel("success change vs clean")
    axes[1].set_title("P2 locked validation: all gates fail (963 items)")
    figure.tight_layout()
    figure.savefig(OUT / "fig2_p2_validation.png", dpi=150)
    plt.close(figure)


def _knowledge_studies() -> list[tuple[str, dict]]:
    studies = []
    for seed in (0, 1, 2):
        path = LOCAL_KNOW / f"seed{seed}" / "evaluation" / "knowledge_result.json"
        if path.exists():
            studies.append((f"0.5B seed {seed}", load(path)))
    path = CC / "know7b_seed0" / "evaluation" / "knowledge_result.json"
    if path.exists():
        studies.append(("7B pilot", load(path)))
    return studies


def fig_knowledge_verification() -> None:
    studies = _knowledge_studies()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    names = [name for name, _ in studies]
    known, unknown = [], []
    for _, result in studies:
        verification = result["verification"]
        means = np.mean(list(verification["variant_means"].values()))
        unseen = np.mean(list(verification["unknown_success"].values()))
        unknown.append(unseen)
        known.append(2 * means - unseen)  # exposure is exactly half per variant
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, known, width=0.4, label="facts the variant learned", color="tab:green")
    axes[0].bar(x + 0.2, unknown, width=0.4, label="facts it never saw", color="tab:red")
    axes[0].axhline(0.2, color="black", linestyle="--", linewidth=0.8)
    axes[0].text(len(names) - 0.4, 0.21, "unexposed-fact limit 0.2", ha="right", fontsize=8)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("answer success")
    axes[0].set_title("Knowledge variants: learned vs unexposed facts")
    axes[0].legend(fontsize=8)

    seven = studies[-1][1]["verification"]
    variants = sorted(seven["variant_means"], key=int)
    xv = np.arange(len(variants))
    axes[1].bar(
        xv - 0.2,
        [2 * seven["variant_means"][v] - seven["unknown_success"][v] for v in variants],
        width=0.4, label="learned facts", color="tab:green",
    )
    axes[1].bar(
        xv + 0.2,
        [seven["unknown_success"][v] for v in variants],
        width=0.4, label="unexposed facts", color="tab:red",
    )
    axes[1].set_xticks(xv, [f"variant {v}" for v in variants])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("7B pilot: per-variant knowledge")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "fig3_knowledge_verification.png", dpi=150)
    plt.close(figure)


def fig_reporter_value() -> None:
    studies = _knowledge_studies()
    names = [name for name, _ in studies]
    t1 = [result["test"]["comparison"]["brier_t1"] for _, result in studies]
    t2 = [result["test"]["comparison"]["brier_t2"] for _, result in studies]
    baseline = [result["test"]["added_value_t2"]["baseline_brier"] for _, result in studies]
    added = [result["test"]["added_value_t2"]["added"] for _, result in studies]
    added_ci = [
        (
            result["test"]["added_value_t2"]["added"]
            - result["test"]["added_value_t2"]["ci"]["ci_low"],
            result["test"]["added_value_t2"]["ci"]["ci_high"]
            - result["test"]["added_value_t2"]["added"],
        )
        for _, result in studies
    ]
    delta = [result["test"]["comparison"]["delta_brier"] for _, result in studies]
    delta_ci = [
        (
            result["test"]["comparison"]["delta_brier"]
            - result["test"]["comparison"]["delta_brier_ci"]["ci_low"],
            result["test"]["comparison"]["delta_brier_ci"]["ci_high"]
            - result["test"]["comparison"]["delta_brier"],
        )
        for _, result in studies
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(names))
    axes[0].bar(x - 0.25, t1, width=0.25, label="T1 pointwise", color="tab:blue")
    axes[0].bar(x, t2, width=0.25, label="T2 paired", color="tab:cyan")
    axes[0].bar(x + 0.25, baseline, width=0.25, label="best baseline", color="tab:gray")
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_ylabel("test Brier (lower is better)")
    axes[0].set_title("Reporter vs baseline on held-out subjects")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(x, added, yerr=np.asarray(added_ci).T, fmt="o", capsize=4,
                     label="report added value (T2 vs baseline)", color="tab:blue")
    axes[1].errorbar(x + 0.08, delta, yerr=np.asarray(delta_ci).T, fmt="s", capsize=4,
                     label="T2 - T1 Brier", color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("Brier difference")
    axes[1].set_title("No added value; paired training does not help")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "fig4_reporter_value.png", dpi=150)
    plt.close(figure)


def fig_readout_probes() -> None:
    probe = load(DIAG / "knowledge_readout_probe2.json")
    order = [
        "mlp_prefix", "question+prefix", "prefix", "question",
        "prefix+rich", "rich",
        "mlp_prefix+rich+hidden_raw", "mlp_hidden_raw", "mlp_prefix+hidden_raw",
        "hidden_raw", "prefix+hidden_raw", "prefix+rich+hidden_raw",
    ]
    labels = []
    values = []
    colors = []
    for key in order:
        if key not in probe:
            continue
        labels.append(key.replace("_", " "))
        values.append(probe[key]["test_squared_error"])
        colors.append("tab:blue" if "hidden" not in key else "tab:red")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(range(len(values))[::-1], values, color=colors)
    axis.set_yticks(range(len(values))[::-1], labels, fontsize=8)
    axis.set_xlabel("test squared error (lower is better)")
    axis.set_title("Readout probes on seed 0 states: hidden features do not help")
    axis.axvline(0.1029, color="tab:green", linestyle="--", linewidth=1.0)
    axis.text(0.104, 0.2, "best probability readout", color="tab:green", fontsize=8, rotation=90)
    figure.tight_layout()
    figure.savefig(OUT / "fig5_readout_probes.png", dpi=150)
    plt.close(figure)


def fig_ablation() -> None:
    ablation = load(DIAG / "knowledge_ablation.json")
    keys = ["question", "question+prefix", "question+hidden", "question+prefix+hidden"]
    labels = [key.replace("+", "\n+") for key in keys]
    t1 = [ablation[key]["t1_brier"] for key in keys]
    t2 = [ablation[key]["t2_brier"] for key in keys]
    baseline = [ablation[key]["added_value"]["baseline_brier"] for key in keys]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(keys))
    axis.bar(x - 0.25, t1, width=0.25, label="T1", color="tab:blue")
    axis.bar(x, t2, width=0.25, label="T2", color="tab:cyan")
    axis.bar(x + 0.25, baseline, width=0.25, label="best baseline", color="tab:gray")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylabel("test Brier")
    axis.set_title("Feature ablation: the seeded hidden block is the poison")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "fig6_feature_ablation.png", dpi=150)
    plt.close(figure)


def main() -> None:
    fig_toy_rl_arms()
    fig_p2_validation()
    fig_knowledge_verification()
    fig_reporter_value()
    fig_readout_probes()
    fig_ablation()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
