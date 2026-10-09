"""Figures for the section 3.3 confirmatory report.

Reads the synced run in runs/section33_confirmatory and writes PNG figures to
report/figures/section33_confirmatory/.  Run from the repository root:

    .venv/bin/python report/make_section33_confirmatory_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = Path("runs/section33_confirmatory")
OUT = Path("report/figures/section33_confirmatory")
OUT.mkdir(parents=True, exist_ok=True)
SLUG = "qwen__qwen2.5_7b_instruct"

TRACKS = ["A", "B"]
TRACK_LABEL = {"A": "Track A (before answer)", "B": "Track B (after answer)"}
MARGIN_AUROC = 0.020
MARGIN_WITHIN = 0.030


def decision(track: str) -> dict:
    return json.loads(
        (RUN / f"evaluation/decision__{SLUG}__track{track}.json").read_text()
    )


def predictions(track: str) -> pd.DataFrame:
    return pd.read_parquet(
        RUN / f"predictions/test_predictions__{SLUG}__track{track}.parquet"
    )


def auroc(frame: pd.DataFrame, score: str) -> float | None:
    labels = frame["label"].to_numpy()
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, frame[score].to_numpy()))


# ---------------------------------------------------------------------------
# Figure 1: primary result
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.2))
positions = np.arange(len(TRACKS))
for index, track in enumerate(TRACKS):
    primary = decision(track)["decision"]
    delta = primary["delta_auc"]
    low, high = primary["delta_auc_interval"]
    ax.errorbar(
        delta, index,
        xerr=[[delta - low], [high - delta]],
        fmt="o", color="navy", capsize=4, markersize=7,
    )
ax.axvline(0.0, color="0.55", lw=1.0)
ax.axvline(MARGIN_AUROC, color="firebrick", ls="--", lw=1.2)
ax.text(MARGIN_AUROC, 1.55, "margin +0.020", color="firebrick", fontsize=8, ha="left")
ax.set_yticks(positions, [TRACK_LABEL[track] for track in TRACKS])
ax.set_xlabel("AUROC difference (B+M minus B)")
ax.set_title("Primary result: both intervals stay below the margin")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "primary.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: method AUROC
# ---------------------------------------------------------------------------
methods = {
    "A": ["B", "B_plus_M", "H_512", "H_768", "H_M_256_256", "H_all_plus_M", "M_all", "P", "V"],
    "B": ["B_after", "B_plus_M_after", "H_512_after", "H_768_after", "H_M_256_256_after",
          "H_all_plus_M_after", "M_all_after", "P_after", "V_after"],
}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
for ax, track in zip(axes, TRACKS):
    data = decision(track)["methods"]
    names = methods[track]
    values = [data[name]["test_auroc"] for name in names]
    colors = ["tab:blue" if name.startswith("B") else "tab:orange" if "M" in name and "H" not in name
              else "tab:green" if name.startswith("H") else "0.6" for name in names]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, [name.replace("_after", "") for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_title(TRACK_LABEL[track])
    ax.set_xlabel("test AUROC")
    ax.grid(axis="x", alpha=0.3)
    for yi, value in zip(y, values):
        ax.text(value + 0.005, yi, f"{value:.3f}", va="center", fontsize=7)
fig.suptitle("Method comparison: MLP features do not beat the baselines")
fig.tight_layout()
fig.savefig(OUT / "methods.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: secondary comparisons
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
large = ["augmented_minus_confidence", "augmented_minus_probability"]
small = ["mlp_minus_hidden", "replication_hidden_plus_mlp_minus_hidden"]
for ax, keys, title, margin in (
    (axes[0], large, "Probe against stated confidence and probability", None),
    (axes[1], small, "MLP features against hidden states", MARGIN_AUROC),
):
    offset = 0.0
    ticks, tick_labels = [], []
    for track in TRACKS:
        secondary = decision(track)["secondary"]
        for key in keys:
            entry = secondary[key]
            delta = entry["delta_auc"]
            low, high = entry["delta_auc_interval"]
            ax.errorbar(delta, offset, xerr=[[delta - low], [high - delta]],
                        fmt="o", capsize=3, markersize=5,
                        color="navy" if track == "A" else "darkgreen")
            ticks.append(offset)
            tick_labels.append(key.replace("_", " "))
            offset -= 1.0
        offset -= 0.4
    if margin is not None:
        ax.axvline(0.0, color="0.55", lw=1.0)
        ax.axvline(margin, color="firebrick", ls="--", lw=1.1)
        ax.axvline(-margin, color="firebrick", ls="--", lw=1.1)
        ax.text(margin, offset + 0.6, "+/- margin", color="firebrick", fontsize=8)
    else:
        ax.axvline(0.0, color="0.55", lw=1.0)
    ax.set_yticks(ticks, tick_labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("AUROC difference")
    ax.grid(axis="x", alpha=0.3)
axes[0].plot([], [], "o", color="navy", label="Track A")
axes[0].plot([], [], "o", color="darkgreen", label="Track B")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "secondary.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: difficulty sensitivity by popularity stratum
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
difficulty_rows = {}
for ax, track in zip(axes, TRACKS):
    frame = predictions(track)
    score_columns = {
        "stated confidence": f"cal_V{'_after' if track == 'B' else ''}",
        "B": f"cal_B{'_after' if track == 'B' else ''}",
        "B+M": f"cal_B_plus_M{'_after' if track == 'B' else ''}",
    }
    strata = sorted(stratum for stratum in frame["stratum"].unique() if stratum != "missing")
    for name, column in score_columns.items():
        values = []
        for stratum in strata:
            block = frame[frame["stratum"] == stratum]
            value = auroc(block, column)
            values.append(value if value is not None else np.nan)
        ax.plot(strata, values, marker="o", label=name)
        difficulty_rows[f"{track}:{name}"] = values
    ax.set_title(TRACK_LABEL[track])
    ax.set_xlabel("popularity stratum (0 = least popular)")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test AUROC")
axes[0].legend(fontsize=8)
fig.suptitle("Difficulty sensitivity: the probe keeps a large easy-hard gap")
fig.tight_layout()
fig.savefig(OUT / "difficulty.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: power
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax = axes[0]
for track in TRACKS:
    power = json.loads(
        (RUN / f"protocol/power__{SLUG}__track{track}.json").read_text()
    )
    grid = [row["n_subjects"] for row in power["rows"]]
    ruled_out = [row["negligible"]["p_useful_improvement_ruled_out"] for row in power["rows"]]
    ax.plot(grid, ruled_out, marker="o", label=f"Track {track}")
ax.axhline(0.9, color="firebrick", ls="--", lw=1.1)
ax.axvline(4000, color="0.55", ls=":", lw=1.0)
ax.set_xlabel("test subjects")
ax.set_ylabel("probability to rule out")
ax.set_title("Power to rule out a useful gain")
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
within = json.loads(
    (RUN / f"protocol/power_within__{SLUG}__trackB.json").read_text()
)
grid = [row["n_subjects"] for row in within["rows"]]
useful = [row["useful"]["p_within_question_gain_supported"] for row in within["rows"]]
ruled_out = [row["negligible"]["p_within_question_gain_ruled_out"] for row in within["rows"]]
ax.plot(grid, useful, marker="o", label="useful gain supported")
ax.plot(grid, ruled_out, marker="s", label="no gain ruled out")
ax.axhline(0.9, color="firebrick", ls="--", lw=1.1)
ax.axvline(4000, color="0.55", ls=":", lw=1.0)
ax.set_xlabel("test subjects")
ax.set_title("Within-question power (Track B)")
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "power.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4b: difficulty sensitivity (residual versus popularity)
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

fig, ax = plt.subplots(figsize=(7.0, 3.4))
methods_for_rho = ["V (stated)", "P", "B", "B+M"]
column_stem = {"V (stated)": "V", "P": "P", "B": "B", "B+M": "B_plus_M"}
suffixes = {"A": "", "B": "_after"}
width = 0.35
positions = np.arange(len(methods_for_rho))
rho_values = {}
for offset, track in zip((-width / 2, width / 2), TRACKS):
    frame = predictions(track)
    popularity = pd.to_numeric(frame["popularity"], errors="coerce").to_numpy()
    values = []
    for name in methods_for_rho:
        column = f"cal_{column_stem[name]}{suffixes[track]}"
        residual = frame["label"].to_numpy() - frame[column].to_numpy()
        mask = np.isfinite(popularity) & np.isfinite(residual)
        values.append(float(spearmanr(popularity[mask], residual[mask]).statistic))
    rho_values[track] = dict(zip(methods_for_rho, values))
    ax.bar(positions + offset, values, width, label=TRACK_LABEL[track])
ax.axhline(0.0, color="0.55", lw=1.0)
ax.set_xticks(positions, methods_for_rho)
ax.set_ylabel("Spearman rho")
ax.set_title("Difficulty sensitivity: residual (outcome minus score) versus popularity")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "difficulty_sensitivity.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Numbers for the report
# ---------------------------------------------------------------------------
summary = {}
for track in TRACKS:
    frame = predictions(track)
    summary[track] = {}
    for name, column in (
        ("stated confidence", f"cal_V{'_after' if track == 'B' else ''}"),
        ("B", f"cal_B{'_after' if track == 'B' else ''}"),
        ("B+M", f"cal_B_plus_M{'_after' if track == 'B' else ''}"),
    ):
        low = frame[frame["stratum"] == "0"]
        high = frame[frame["stratum"] == "4"]
        summary[track][name] = {
            "auroc_low_popularity": auroc(low, column),
            "auroc_high_popularity": auroc(high, column),
        }
(Path("report/figures/section33_confirmatory/difficulty_summary.json")).write_text(
    json.dumps(summary, indent=2)
)
print(json.dumps(summary, indent=2))
print("figures written to", OUT)
