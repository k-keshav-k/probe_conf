"""Figures for the selective-metacognition toy report.

    .venv/bin/python report/make_selective_metacognition_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

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

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

ax = axes[0]
clean = seed_values(e1, "C", ("fresh", "clean", "top1")) - seed_values(e1, "B", ("fresh", "clean", "top1"))
swap = seed_values(e1, "C", ("fresh", "target_swap_valid", "top1")) - seed_values(
    e1, "B", ("fresh", "target_swap_valid", "top1")
)
rng = np.random.default_rng(0)
for offset, values, color, label in ((0, clean, "#c44e52", "clean top-1"), (1, swap, "#4c72b0", "valid-swap top-1")):
    values = 100 * values
    jitter = rng.uniform(-0.08, 0.08, len(values))
    ax.scatter(np.full(len(values), offset) + jitter, values, color=color, s=28, zorder=3)
    ax.hlines(values.mean(), offset - 0.25, offset + 0.25, color=color, linewidth=2.4, zorder=4)
    ax.text(offset, ax.get_ylim()[1], "", ha="center")
ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks([0, 1], ["Clean\nquestions", "Valid fact\nswaps"])
ax.set_ylabel("C - B accuracy (percentage points)")
ax.set_title("E1: better retrieval on new assignments")
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
means = [seed_values(e2, arm, ("reporting_test", "squared_probability_error")).mean() for arm in ARMS]
errors = [seed_values(e2, arm, ("reporting_test", "squared_probability_error")).std() for arm in ARMS]
ax.bar(range(len(ARMS)), means, yerr=errors, color=[COLORS[arm] for arm in ARMS], capsize=4)
ax.set_xticks(range(len(ARMS)), ARMS)
ax.set_ylabel("Probability error (lower is better)")
ax.set_title("E2: C and D have lower probability error")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fig1_e1_e2.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: E3 prospective increments and the probe diagnostic
# ---------------------------------------------------------------------------
e3 = load(RESULTS / "e3.json")
probe = load(LOCAL / "e3_probe_diagnostic.json")

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

ax = axes[0]
arms_e3 = ("A", "B", "C", "E")
width = 0.38
test = [seed_values(e3, arm, ("prospective_test", "incremental_brier")) for arm in arms_e3]
held = [seed_values(e3, arm, ("prospective_heldout", "incremental_brier")) for arm in arms_e3]
positions = np.arange(len(arms_e3))
ax.bar(positions - width / 2, [v.mean() for v in test], width, yerr=[v.std() for v in test],
       color="#4c72b0", capsize=4, label="Previously used strengths")
ax.bar(positions + width / 2, [v.mean() for v in held], width, yerr=[v.std() for v in held],
       color="#c44e52", capsize=4, label="Unseen strengths")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(positions, arms_e3)
ax.set_ylabel("Brier improvement from report")
ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax.set_title("E3: very little added value")
ax.legend(fontsize=8.5)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
predictors = ("baseline", "ridge_hidden_oracle", "mlp_hidden_oracle", "mlp_hidden_outcomes")
labels = ("Probability\nbaseline", "Linear\nprobe", "Neural\nexact", "Neural\nsampled")
means, errors = [], []
for predictor in predictors:
    values = np.asarray([probe["arms"][arm][predictor]["brier"] for arm in probe["arms"]])
    means.append(values.mean())
    errors.append(values.std())
ax.bar(range(len(predictors)), means, color=["#55a868", "#999999", "#4c72b0", "#c44e52"], alpha=0.75)
for i, predictor in enumerate(predictors):
    values = [probe["arms"][arm][predictor]["brier"] for arm in probe["arms"]]
    ax.scatter(np.arange(len(values)) * 0.12 + i - 0.12, values, color="#263442", s=18, zorder=3)
ax.set_xticks(range(len(predictors)), labels, fontsize=9)
ax.set_ylabel("Brier error (lower is better)")
ax.set_title("Tested probes trail the baseline\nOne seed; dots are arms B/C/E")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fig2_e3_probe.png", dpi=200)
plt.close(fig)

print(f"wrote {OUT / 'fig1_e1_e2.png'}")
print(f"wrote {OUT / 'fig2_e3_probe.png'}")
