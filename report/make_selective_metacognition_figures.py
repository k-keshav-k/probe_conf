"""Figures for the selective-metacognition toy report.

    .venv/bin/python report/make_selective_metacognition_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "section33_v1" / "selective_metacognition"
LOCAL = ROOT / "runs" / "selective_metacognition"
OUT = ROOT / "report" / "figures" / "selective_metacognition"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ("A", "B", "C", "D", "E")
COLORS = {"A": "#999999", "B": "#4c72b0", "C": "#c44e52", "D": "#8172b2", "E": "#55a868"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def seed_values(report: dict, arm: str, path: tuple[str, ...]) -> np.ndarray:
    values = []
    for seed in sorted(report["per_seed"]):
        block = report["per_seed"][seed][arm]
        for key in path:
            block = block[key]
        values.append(block)
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# Figure 1: E1 transfer and E2 squared probability error
# ---------------------------------------------------------------------------
e1 = load(RESULTS / "e1.json")
e2 = load(RESULTS / "e2.json")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

ax = axes[0]
clean = seed_values(e1, "C", ("fresh", "clean", "top1")) - seed_values(e1, "B", ("fresh", "clean", "top1"))
swap = seed_values(e1, "C", ("fresh", "target_swap_valid", "top1")) - seed_values(
    e1, "B", ("fresh", "target_swap_valid", "top1")
)
rng = np.random.default_rng(0)
for offset, values, color, label in ((0, clean, "#c44e52", "clean top-1"), (1, swap, "#4c72b0", "valid-swap top-1")):
    jitter = rng.uniform(-0.08, 0.08, len(values))
    ax.scatter(np.full(len(values), offset) + jitter, values, color=color, s=28, zorder=3)
    ax.hlines(values.mean(), offset - 0.25, offset + 0.25, color=color, linewidth=2.4, zorder=4)
    ax.text(offset, ax.get_ylim()[1], "", ha="center")
ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks([0, 1], ["clean top-1", "valid-swap top-1"])
ax.set_ylabel("C - B on fresh bindings")
ax.set_title("E1: selectivity training transfers (5 seeds)", fontsize=10)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
means = [seed_values(e2, arm, ("reporting_test", "squared_probability_error")).mean() for arm in ARMS]
errors = [seed_values(e2, arm, ("reporting_test", "squared_probability_error")).std() for arm in ARMS]
ax.bar(range(len(ARMS)), means, yerr=errors, color=[COLORS[arm] for arm in ARMS], capsize=4)
ax.set_xticks(range(len(ARMS)), ARMS)
ax.set_ylabel("squared error vs true success probability")
ax.set_title("E2: C's report is closest to the true probability", fontsize=10)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fig1_e1_e2.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: E3 prospective increments and the probe diagnostic
# ---------------------------------------------------------------------------
e3 = load(RESULTS / "e3.json")
probe = load(LOCAL / "e3_probe_diagnostic.json")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

ax = axes[0]
arms_e3 = ("A", "B", "C", "E")
width = 0.38
test = [seed_values(e3, arm, ("prospective_test", "incremental_brier")) for arm in arms_e3]
held = [seed_values(e3, arm, ("prospective_heldout", "incremental_brier")) for arm in arms_e3]
positions = np.arange(len(arms_e3))
ax.bar(positions - width / 2, [v.mean() * 1e4 for v in test], width, yerr=[v.std() * 1e4 for v in test],
       color="#4c72b0", capsize=4, label="training strengths")
ax.bar(positions + width / 2, [v.mean() * 1e4 for v in held], width, yerr=[v.std() * 1e4 for v in held],
       color="#c44e52", capsize=4, label="held-out strengths")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(positions, arms_e3)
ax.set_ylabel("incremental Brier (x 1e-4)")
ax.set_title("E3: report adds ~0 over clean probabilities + description", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
predictors = ("baseline", "ridge_hidden_oracle", "mlp_hidden_oracle", "mlp_hidden_outcomes")
labels = ("baseline\n(clean probs\n+ description)", "linear probe\non hidden\n(oracle target)",
          "MLP probe\non hidden\n(oracle target)", "MLP probe\non hidden\n(outcome target)")
means, errors = [], []
for predictor in predictors:
    values = np.asarray([probe["arms"][arm][predictor]["brier"] for arm in probe["arms"]])
    means.append(values.mean())
    errors.append(values.std())
ax.bar(range(len(predictors)), means, yerr=errors, color=["#55a868", "#999999", "#4c72b0", "#c44e52"], capsize=4)
ax.set_xticks(range(len(predictors)), labels, fontsize=7.5)
ax.set_ylabel("Brier vs post-intervention outcomes")
ax.set_title("Probe diagnostic: no hidden-state readout beats the baseline", fontsize=10)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fig2_e3_probe.png", dpi=200)
plt.close(fig)

print(f"wrote {OUT / 'fig1_e1_e2.png'}")
print(f"wrote {OUT / 'fig2_e3_probe.png'}")
