"""Dashboard figure for report/PROJECT_SUMMARY.md (one page, six panels).

Everything is read from saved artifacts; no computation beyond plotting.
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

LOCAL = ROOT / "runs" / "causal_confidence" / "cc_v1" / "knowledge"
CLUSTER_CC = ROOT / "runs" / "section33_v1" / "causal_confidence" / "cc_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def knowledge_studies() -> list[tuple[str, dict]]:
    studies = []
    for seed in (0, 1, 2):
        path = LOCAL / f"seed{seed}" / "evaluation" / "knowledge_result.json"
        if path.exists():
            studies.append((f"0.5B s{seed}", load(path)))
    for seed in (0, 1):
        path = CLUSTER_CC / f"know7b_seed{seed}" / "evaluation" / "knowledge_result.json"
        if path.exists():
            studies.append((f"7B s{seed}", load(path)))
    return studies


def panel_added_value(axis, studies):
    names = [name for name, _ in studies]
    added = [result["test"]["added_value_t2"]["added"] for _, result in studies]
    errors = [
        (
            result["test"]["added_value_t2"]["added"]
            - result["test"]["added_value_t2"]["ci"]["ci_low"],
            result["test"]["added_value_t2"]["ci"]["ci_high"]
            - result["test"]["added_value_t2"]["added"],
        )
        for _, result in studies
    ]
    axis.bar(names, added, yerr=np.asarray(errors).T, capsize=4, color="tab:green")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Added value vs probability baseline\n(5/5 positive, intervals above 0)")
    axis.set_ylabel("Brier improvement")


def panel_pair_delta(axis, studies):
    names = [name for name, _ in studies]
    delta = [result["test"]["comparison"]["delta_brier"] for _, result in studies]
    axis.bar(names, delta, color="tab:orange")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Paired objective T2 - T1\n(negative in every study)")
    axis.set_ylabel("Brier difference")


def panel_transfer(axis, studies):
    names = [name for name, _ in studies]
    added = [result["transfer_unseen_variant"]["added_value_t2"]["added"] for _, result in studies]
    axis.bar(names, added, color="tab:red")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Transfer to unseen knowledge variant\n(negative where measured)")
    axis.set_ylabel("Brier improvement")


def panel_native(axis):
    diagnostics = LOCAL / "diagnostics"
    labels, emitted_corr, expected_corr = [], [], []
    for seed in (0, 1, 2):
        path = diagnostics / f"native_sweep_{seed}.json"
        if not path.exists():
            continue
        data = load(path)
        for arm, result in sorted(data["arms"].items()):
            labels.append(f"s{seed} {arm}")
            emitted_corr.append(result["within_question_correlation"] or 0.0)
            expected_corr.append(result["expected_within_question_correlation"] or 0.0)
    x = np.arange(len(labels))
    axis.bar(x - 0.2, emitted_corr, width=0.4, label="emitted number", color="tab:gray")
    axis.bar(x + 0.2, expected_corr, width=0.4, label="expected (distribution)", color="tab:blue")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, fontsize=7)
    axis.set_title("Native arms: tracking hidden by greedy emission")
    axis.set_ylabel("within-question correlation")
    axis.legend(fontsize=7)


def panel_p2(axis):
    validation = load(CLUSTER_CC / "p2c" / "interventions" / "p2_validation.json")
    effects = validation["effects"]
    names = ["damage", "target-random", "restoration"]
    values = [effects["damage_strong"], effects["target_minus_random"], effects["restoration"]["recovered"]]
    thresholds = [0.05, 0.03, 0.5]
    x = np.arange(len(names))
    axis.bar(x, values, color="tab:red")
    for index, threshold in enumerate(thresholds):
        axis.plot([index - 0.4, index + 0.4], [threshold, threshold], color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, names)
    axis.set_title("P2 attenuation gates\n(all fail; branch closed)")
    axis.set_ylabel("effect (dashed = required)")


def panel_audit(axis):
    audit = load(LOCAL / "diagnostics" / "knowledge_readout_audit.json")
    names = ["baseline", "hidden probe"]
    values = [audit["baseline"]["test_brier"], audit["probe_ridge"]["reproduction_brier"]]
    axis.bar(names, values, color=["tab:gray", "tab:blue"])
    axis.set_title(
        "Hidden state reproduces the baseline\n"
        f"reproduction MSE {audit['probe_ridge']['reproduction_mse']:.5f} (gate 0.001)"
    )
    axis.set_ylabel("Brier")


def main() -> None:
    studies = knowledge_studies()
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    panel_added_value(axes[0][0], studies)
    panel_pair_delta(axes[0][1], studies)
    panel_transfer(axes[0][2], studies)
    panel_native(axes[1][0])
    panel_p2(axes[1][1])
    panel_audit(axes[1][2])
    figure.suptitle("Confidence reporting: where the evidence stands (16 Oct 2026)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUT / "fig0_summary_dashboard.png", dpi=150)
    plt.close(figure)
    print("written", OUT / "fig0_summary_dashboard.png")


if __name__ == "__main__":
    main()
