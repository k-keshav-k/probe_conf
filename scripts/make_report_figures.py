"""Generate clean, report-ready figures for the pilot.

Outputs to report/figures/:
  fig0_pipeline.png        what the experiment does (schematic)
  fig1_auroc_bars.png      humans-readable AUROC comparison with CIs
  fig2_layer_curves.png    where in the network the signal lives
  fig3_within_bins.png     does the signal survive difficulty control?
  fig4_overconfidence.png  accuracy vs verbalized confidence across popularity
  fig5_combined_delta.png  bootstrap gain from combining hidden + MLP features

Run: .venv/bin/python scripts/make_report_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)

C = {
    "confidence": "#9e9e9e",
    "surface": "#C44E52",
    "prob": "#8172B3",
    "mlp": "#DD8452",
    "hidden": "#4C72B0",
    "combined": "#55A868",
}

NUMBERS: dict[str, float] = {}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    lab = pd.read_csv(RES / "pilot7b_labels.csv")
    feat = pd.read_parquet(RES / "pilot7b_features.parquet").set_index("id").loc[lab["id"].values]
    oof_f = np.load(RES / "pilot7b_frac_extra_oof.npz")
    oof_g = np.load(RES / "pilot7b_greedy_extra_oof.npz")
    return lab, feat, oof_f, oof_g


def bootstrap_ci(y, score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_deltas(y, a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return np.array(out)


# ---------------------------------------------------------------------------
# fig 0: schematic
# ---------------------------------------------------------------------------
def fig0():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1FB", ec="#4C72B0", fs=10.5, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                fc=fc, ec=ec, lw=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color=color))

    box(0.1, 1.25, 1.9, 0.9, "PopQA question\n(e.g. \"Who is the\nfather of X?\")", bold=True)
    box(2.5, 1.0, 2.0, 1.4, "Qwen2.5-7B\nInstruct\n(answers, confidence,\nprompt + gold token)", fc="#F3EAF9", ec="#8172B3", bold=True)

    box(5.3, 2.25, 2.6, 0.85, "What the model says:\nconfidence 1-100", fc="#F2F2F2", ec="#9e9e9e")
    box(5.3, 1.2, 2.6, 0.85, "What is inside:\nhidden states + MLP\nselection features", fc="#FBEFE6", ec="#DD8452")
    box(5.3, 0.15, 2.6, 0.85, "Correct answers:\n30 answers per question,\nfraction correct", fc="#EAF6EE", ec="#55A868")

    box(8.4, 1.2, 2.4, 1.9,
        "Probes predict a\nquestion-level label:\nfraction correct >= 0.5\n\nOffline experiment;\nfeatures use gold data\n(reported as AUROC)", fc="#FFF9E6", ec="#B8A24A", bold=False)

    arrow(2.0, 1.7, 2.5, 1.7)
    arrow(4.5, 2.45, 5.3, 2.65)
    arrow(4.5, 1.7, 5.3, 1.6)
    arrow(4.5, 1.05, 5.3, 0.55)
    arrow(7.9, 2.65, 8.4, 2.2)
    arrow(7.9, 1.6, 8.4, 1.7)
    arrow(7.9, 0.55, 8.4, 1.2)
    ax.set_title("The experiment at a glance", fontsize=13)
    fig.savefig(OUT / "fig0_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 1: AUROC bars
# ---------------------------------------------------------------------------
def fig1(lab, feat, oof):
    y = (lab.frac_correct.values >= 0.5).astype(int)
    p_gold = feat.p_gold.values
    pg_lo, pg_hi = bootstrap_ci(y, p_gold)
    NUMBERS.update(
        p_gold_auroc=float(roc_auc_score(y, p_gold)),
        p_gold_ci_lo=pg_lo,
        p_gold_ci_hi=pg_hi,
    )

    ex = pd.read_csv(RES / "pilot7b_frac_extra.csv").set_index("feature_set")
    entries = [
        ("Confidence logistic probe\n(1 feature; failures clipped to 0)", ex.loc["confidence"], C["confidence"]),
        ("Metadata probe\n(9 features; gold metadata included)", ex.loc["surface"], C["surface"]),
        ("Gold first-token probability\n(one token, not the complete answer)", None, C["prob"]),
        ("Probability probe\n(5 features; gold token included)", ex.loc["prob"], C["prob"]),
        ("MLP summaries\n(last prompt, layer 2; 17 features)", ex.loc["Msum_last_l2"], C["mlp"]),
        ("Hidden state\n(gold-token position, layer 18)", ex.loc["H_ans_l18"], C["hidden"]),
        ("Hidden + all MLP summaries\n(4,536 features; gold-dependent)", ex.loc["H18+Msum_all"], C["combined"]),
    ]

    labels, vals, los, his, colors = [], [], [], [], []
    for label, row, color in entries:
        labels.append(label)
        colors.append(color)
        if row is None:
            vals.append(roc_auc_score(y, p_gold))
            los.append(pg_lo)
            his.append(pg_hi)
        else:
            vals.append(row.auroc_oof)
            los.append(row.ci_lo)
            his.append(row.ci_hi)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    pos = np.arange(len(vals))
    xerr = np.array([[v - lo for v, lo in zip(vals, los)], [hi - v for v, hi in zip(vals, his)]])
    ax.barh(pos, vals, color=colors, alpha=0.9, height=0.62)
    ax.errorbar(vals, pos, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=1.4, capsize=4)
    ax.axvline(0.5, color="#888888", ls="--", lw=1.4)
    ax.text(0.505, -0.52, "random guessing", color="#666666", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 0.014, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0.45, 0.99)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlabel("AUROC for the label: fraction correct >= 0.5")
    ax.set_title("Saved offline scores: internal probes rank the label better")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig1_auroc_bars.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 2: layer curves
# ---------------------------------------------------------------------------
def fig2():
    r = pd.read_csv(RES / "pilot7b_probe_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, layer_kind, title in [
        (axes[0], "ans", "Features at the supplied gold-token position"),
        (axes[1], "last", "Features read at the last prompt token"),
    ]:
        h = r[(r.family == f"H_{layer_kind}")].sort_values("layer")
        m = r[(r.family == f"Msum_{layer_kind}")].sort_values("layer")
        ax.plot(h.layer, h.auroc_oof, "-o", ms=4, color=C["hidden"], label="Hidden states")
        ax.plot(m.layer, m.auroc_oof, "-s", ms=4, color=C["mlp"], label="MLP summary features")
        if layer_kind == "ans":
            peak = h.loc[h.auroc_oof.idxmax()]
            ax.annotate(
                f"peak at layer {int(peak.layer)}",
                xy=(peak.layer, peak.auroc_oof), xytext=(peak.layer - 11, 0.905),
                fontsize=9, color=C["hidden"],
                arrowprops=dict(arrowstyle="-", color=C["hidden"], lw=1),
            )
        else:
            ax.annotate(
                "informative from\nthe first layer",
                xy=(0.2, 0.86), xytext=(3, 0.91),
                fontsize=9, color=C["mlp"],
                arrowprops=dict(arrowstyle="-", color=C["mlp"], lw=1),
            )
        ax.axhline(0.5, color="#888888", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("transformer layer (0 = first)")
        ax.set_ylim(0.48, 0.94)
    axes[0].set_ylabel("AUROC (pooled, subject-disjoint CV)")
    axes[0].legend(loc="lower right", fontsize=10)
    fig.suptitle("Layer-wise AUROC; both MLP summary families include gold-dependent features",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_layer_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 3: within popularity bins
# ---------------------------------------------------------------------------
def fig3(lab, feat):
    wb = pd.read_csv(RES / "pilot7b_frac_extra_withinbin.csv").set_index("score")
    show = [
        ("confidence", "Verbalized confidence", C["confidence"]),
        ("surface", "Metadata probe", C["surface"]),
        ("Msum_last_l2", "MLP features", C["mlp"]),
        ("H_ans_l18", "Hidden states", C["hidden"]),
        ("H18+Msum_all", "Hidden + MLP", C["combined"]),
    ]
    bins = [f"bin{i}" for i in range(5)]
    x = np.arange(len(bins))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for j, (key, label, color) in enumerate(show):
        vals = wb.loc[key, bins].values.astype(float)
        ax.bar(x + (j - 2) * width, vals, width, label=label, color=color, alpha=0.9)
    ax.axhline(0.5, color="#888888", ls="--", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Q1\n(least popular)", "Q2", "Q3", "Q4", "Q5\n(most popular)"]
    )
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC within popularity group")
    ax.set_xlabel("Popularity group (from Wikipedia monthly page views)", labelpad=8)
    ax.set_title("Ranking within popularity bins; difficulty can still vary inside a bin")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=9)
    fig.savefig(OUT / "fig3_within_bins.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 4: overconfidence
# ---------------------------------------------------------------------------
def fig4(lab, feat):
    frac = lab.frac_correct.values
    conf = lab.confidence.values
    valid = (conf >= 1) & (conf <= 100)
    pop = np.log10(feat.s_pop.values + 1)
    q = pd.qcut(pop, 5, labels=False)
    acc = [frac[q == i].mean() * 100 for i in range(5)]
    mc = [conf[(q == i) & valid].mean() for i in range(5)]
    NUMBERS.update(quintile_accuracy=[float(a) for a in acc], quintile_confidence=[float(c) for c in mc])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = np.arange(5)
    ax.bar(x - 0.19, acc, 0.36, color=C["combined"], label="Mean fraction correct")
    ax.bar(x + 0.19, mc, 0.36, color=C["confidence"], label="Mean valid confidence")
    for xi, a, c in zip(x, acc, mc):
        ax.text(xi - 0.19, a + 2.5, f"{a:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + 0.19, c - 9, f"{c:.0f}%", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percent")
    ax.set_title("Sampling accuracy across popularity groups")
    ax.set_xlabel("Popularity group")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    edges = [0, 60, 80, 90, 95, 100]
    centers, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        centers.append(float(conf[m].mean()))
        accs.append(frac[m].mean() * 100)
        ns.append(int(m.sum()))
    ax.plot([0, 100], [0, 100], "--", color="#888888", lw=1.2, label="Equal percentages")
    ax.scatter(centers, accs, s=[max(30, n / 3) for n in ns], color=C["surface"], zorder=3)
    ax.plot(centers, accs, color=C["surface"], lw=2, alpha=0.7)
    for c, a, n in zip(centers, accs, ns):
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(10, 8), fontsize=8)
    ax.set_xlim(20, 102)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Mean fraction correct (%)")
    ax.set_title("Confidence bands and sampling accuracy")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(35, 40, "91.1 mean valid confidence\n17.0% mean sampled accuracy\n3 failed confidence parses omitted", fontsize=9, color="#555555")

    fig.suptitle("High confidence and low sampled accuracy; the prompts differ",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_overconfidence.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 5: combined gain bootstrap
# ---------------------------------------------------------------------------
def fig5(lab, oof_f, oof_g):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for oof, label, color in [
        (oof_f, "label: fraction correct >= 0.5", C["hidden"]),
        (oof_g, "label: greedy answer correct", C["combined"]),
    ]:
        if "H18+Msum_all" not in oof:
            continue
        y = (lab.frac_correct.values >= 0.5).astype(int) if "frac" in label else lab.greedy_correct.values.astype(int)
        d = bootstrap_deltas(y, oof["H18+Msum_all"], oof["H_ans_l18"])
        ax.hist(d, bins=40, alpha=0.55, color=color, label=f"{label}  (mean {d.mean():+.3f})")
        NUMBERS[f"delta_mean_{'frac' if 'frac' in label else 'greedy'}"] = float(d.mean())
        NUMBERS[f"delta_ci_{'frac' if 'frac' in label else 'greedy'}"] = [
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        ]
    ax.axvline(0, color="#333333", lw=1.4, ls="--")
    ax.set_xlabel("AUROC increase from the MLP features")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title("The combination of hidden state and MLP features increases the AUROC")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig5_combined_delta.png")
    plt.close(fig)


def main():
    lab, feat, oof_f, oof_g = load()
    fig0()
    fig1(lab, feat, oof_f)
    fig2()
    fig3(lab, feat)
    fig4(lab, feat)
    fig5(lab, oof_f, oof_g)
    with open(RES / "report_numbers.json", "w") as f:
        json.dump(NUMBERS, f, indent=2)
    print("wrote figures to", OUT)
    print(json.dumps(NUMBERS, indent=2))


if __name__ == "__main__":
    main()
