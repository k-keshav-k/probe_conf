"""Evaluate probes: hidden states vs MLP interaction features.

Loads labels + features + raw tensors collected by scripts/collect.py, runs
subject-disjoint grouped CV for each feature set / layer, and writes:
  results/<tag>_probe_results.csv     all metrics
  results/<tag>_report.md             summary + verdict against H1-H4
  results/figures/*.png               layer curves, comparisons, diagnostics

Usage:
    python scripts/probe_eval.py --tag pilot --k 30
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    FIG_DIR,
    PERSON_RELATIONS,
    RESULTS_DIR,
)
from scripts.collect import SUMMARY_STATS  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SEED = 0
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_SPLITS = 5
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw(tag: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(RESULTS_DIR / "raw" / f"{tag}_batch*.npz")))
    if not files:
        raise SystemExit(f"no raw feature files for tag={tag}")
    ids, h_last, h_ans, gate, sel, down = [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        ids.append(z["ids"])
        h_last.append(z["h_last"])
        h_ans.append(z["h_ans"])
        gate.append(z["gate_pre"])
        sel.append(z["sel"])
        down.append(z["down"])
    return {
        "ids": np.concatenate(ids),
        "h_last": np.concatenate(h_last, axis=1).transpose(1, 0, 2),  # (n, L+1, d)
        "h_ans": np.concatenate(h_ans, axis=1).transpose(1, 0, 2),
        "gate_pre": np.concatenate(gate),  # (n, L, 2, d_ff)
        "sel": np.concatenate(sel),
        "down": np.concatenate(down),
    }


def align(feats: pd.DataFrame, labels: pd.DataFrame, raw: dict) -> tuple[pd.DataFrame, dict]:
    """Reorder everything to features order; return (merged df, raw aligned)."""
    if not feats["id"].is_unique or not labels["id"].is_unique:
        raise ValueError("feature and label IDs must be unique")
    raw_ids = np.asarray(raw["ids"])
    if len(np.unique(raw_ids)) != len(raw_ids):
        raise ValueError("raw IDs must be unique")
    target = feats["id"].to_numpy()
    if set(target) != set(labels["id"]) or set(target) != set(raw_ids):
        raise ValueError("feature, label, and raw ID sets must match")
    raw_order = {int(i): k for k, i in enumerate(raw_ids)}
    labels = labels.set_index("id").loc[feats["id"].values].reset_index()
    df = feats.merge(labels, on="id", suffixes=("", "_lab"), sort=False, validate="one_to_one")
    indices = [raw_order[int(i)] for i in target]
    raw_aligned = {k: v[indices] for k, v in raw.items() if k != "ids"}
    return df, raw_aligned


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------
def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logreg",
    seed: int = SEED,
    pca_dim: int | None = None,
) -> dict:
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    oof = np.full(len(y), np.nan)
    fold_aurocs = []
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        # Preprocessing is unsupervised: fit once on the training fold, then
        # reuse for C selection and the final fit (avoids ~90 PCA refits).
        steps = [StandardScaler()]
        if pca_dim:
            steps.append(PCA(n_components=min(pca_dim, len(tr) - 1), random_state=seed))
        pre = make_pipeline(*steps)
        X_tr = pre.fit_transform(X[tr])
        X_te = pre.transform(X[te])

        if kind == "logreg":
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            best_c, best_s = None, -np.inf
            for c in C_GRID:
                scores = []
                for itr, ite in inner.split(X_tr, y[tr], groups[tr]):
                    clf = LogisticRegression(C=c, max_iter=3000)
                    clf.fit(X_tr[itr], y[tr][itr])
                    scores.append(roc_auc_score(y[tr][ite], clf.predict_proba(X_tr[ite])[:, 1]))
                s = float(np.mean(scores))
                if s > best_s:
                    best_c, best_s = c, s
            clf = LogisticRegression(C=best_c, max_iter=3000)
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            )
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)[:, 1]
        fold_aurocs.append(roc_auc_score(y[te], oof[te]))

    mask = ~np.isnan(oof)
    auroc_oof = roc_auc_score(y[mask], oof[mask])
    boot = []
    rng = np.random.default_rng(seed)
    idx_all = np.where(mask)[0]
    for _ in range(N_BOOT):
        b = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        boot.append(roc_auc_score(y[b], oof[b]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)
    return {
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "auroc_oof": float(auroc_oof),
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "n_features": int(X.shape[1]),
        "oof": oof,
    }


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = SEED):
    mask = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(deltas)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def stage2_difficulty(df, y, frac, oofs, res_records, tag):
    """Within-popularity-bin AUROCs + residual diagnostic. Returns markdown lines."""
    pop = np.log10(df["s_pop"].values.astype(float) + 1.0)
    bins = pd.qcut(pop, 5, labels=False, duplicates="drop")

    def best_name(fam):
        sub = [r for r in res_records if r.get("family") == fam]
        return max(sub, key=lambda r: r["auroc_oof"])["feature_set"] if sub else None

    score_names = {
        "surface (difficulty)": "surface",
        "verbalized confidence": "confidence",
        "model probs": "prob",
        "hidden states": best_name("H_ans") or best_name("H_last"),
        "MLP summaries": best_name("Msum_ans") or best_name("Msum_last"),
    }
    score_names = {k: v for k, v in score_names.items() if v in oofs}

    rows = []
    for label, name in score_names.items():
        oof = oofs[name].astype(float)
        mask = np.isfinite(oof)
        yy, oo = y[mask], oof[mask]
        row = {"score": label, "feature_set": name, "overall_auroc": roc_auc_score(yy, oo)}
        for b in np.unique(bins):
            m = (bins == b) & mask
            row[f"bin{int(b)}_auroc"] = (
                roc_auc_score(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan
            )
        rows.append(row)
    s2 = pd.DataFrame(rows)
    s2.to_csv(RESULTS_DIR / f"{tag}_stage2.csv", index=False)

    # per-relation AUROCs for the same scores
    rel_rows = []
    for label, name in score_names.items():
        oof = oofs[name]
        row = {"score": label}
        for rel in sorted(df["prop"].unique()):
            m = ((df["prop"] == rel).values) & np.isfinite(oof)
            row[rel] = roc_auc_score(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan
        rel_rows.append(row)
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(RESULTS_DIR / f"{tag}_per_relation.csv", index=False)

    # Diagnostic gap, not a fitted regression residual: surface OOF predicts a
    # binary target, whereas frac_correct is a continuous sample fraction.
    ps = oofs["surface"]
    resid = frac - ps
    res_rows = []
    for label, name in score_names.items():
        oof = oofs[name].astype(float)
        mask = np.isfinite(oof) & np.isfinite(resid)
        rho = spearmanr(oof[mask], resid[mask]).statistic
        res_rows.append({"score": label, "spearman_with_residual": float(rho)})
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(RESULTS_DIR / f"{tag}_residuals.csv", index=False)

    # figure: AUROC by bin
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_cols = [c for c in s2.columns if c.startswith("bin")]
    x = np.arange(len(bin_cols))
    for _, r in s2.iterrows():
        ax.plot(x, r[bin_cols].values.astype(float), marker="o", label=r["score"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_auroc", "") for c in bin_cols])
    ax.set_ylabel("AUROC within popularity bin")
    ax.set_xlabel("s_pop quintile (0=least popular)")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_stage2_bins.png", dpi=150)
    plt.close(fig)

    lines = ["", "## Stage 2: difficulty control (s_pop quintiles)", ""]
    lines.append(s2.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("### Per-relation AUROC")
    lines.append("")
    lines.append("```")
    lines.append(rel_df.to_string(index=False, float_format="%.3f"))
    lines.append("```")
    lines.append("")
    lines.append("Diagnostic gap (not a fitted regression residual): Spearman with `frac_correct - surface_OOF`:")
    lines.append("")
    lines.append(res_df.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    return lines, s2, res_df


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
def build_metadata_features(df: pd.DataFrame) -> np.ndarray:
    cols = []
    cols.append(np.log10(df["s_pop"].values.astype(float) + 1.0))
    cols.append(np.log10(df["o_pop"].values.astype(float).clip(min=0) + 1.0))
    for rel in PERSON_RELATIONS:
        cols.append((df["prop"].values == rel).astype(float))
    cols.append(df["n_aliases"].values.astype(float))
    cols.append(df["prompt_tokens"].values.astype(float))
    cols.append(df["subject_tokens"].values.astype(float))
    return np.stack(cols, axis=1)


def build_prob_features(df: pd.DataFrame) -> np.ndarray:
    return df[["p_gold", "p_entropy", "p_top1", "p_margin", "p_gold_rank"]].values.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--label", choices=["frac", "greedy"], default="frac")
    ap.add_argument("--raw-probes", action="store_true", help="also probe raw MLP vectors")
    ap.add_argument("--neurotic", action="store_true")
    args = ap.parse_args()

    out_tag = args.tag if args.label == "frac" else f"{args.tag}_{args.label}"

    feats = pd.read_parquet(RESULTS_DIR / f"{args.tag}_features.parquet")
    labels_path = RESULTS_DIR / f"{args.tag}_labels.csv"
    labels = pd.read_csv(labels_path)
    raw = load_raw(args.tag)
    df, raw = align(feats, labels, raw)
    n = len(df)
    L = raw["h_last"].shape[1] - 1  # number of transformer layers (exclude embeddings)
    frac = df["frac_correct"].values.astype(float)

    if args.label == "greedy":
        y = df["greedy_correct"].values.astype(int)
    else:
        y = (frac >= 0.5).astype(int)
    y_greedy = df["greedy_correct"].values.astype(int)
    groups = df["subj_id"].values
    print(f"n={n}, positive rate (frac>=0.5)={y.mean():.3f}, greedy acc={y_greedy.mean():.3f}")
    print(f"mean confidence={np.clip(df['confidence'].values, 0, 100).mean():.1f}")
    print(f"layers: {L} (indices 0 to {L-1})")

    meta_X = build_metadata_features(df)
    prob_X = build_prob_features(df)
    conf_X = np.clip(df["confidence"].values.astype(float), 0, 100).reshape(-1, 1)

    feature_sets: dict[str, np.ndarray] = {
        "surface": meta_X,
        "prob": prob_X,
        "confidence": conf_X,
        "confidence+surface": np.hstack([conf_X, meta_X]),
    }
    for l in range(L):
        feature_sets[f"H_last_l{l}"] = raw["h_last"][:, l + 1, :]
        feature_sets[f"H_ans_l{l}"] = raw["h_ans"][:, l + 1, :]
        stat_cols = [f"m_last_l{l}_{s}" for s in SUMMARY_STATS]
        feature_sets[f"Msum_last_l{l}"] = df[stat_cols].values.astype(float)
        stat_cols = [f"m_ans_l{l}_{s}" for s in SUMMARY_STATS]
        feature_sets[f"Msum_ans_l{l}"] = df[stat_cols].values.astype(float)

    feature_sets["H_last_all"] = raw["h_last"].reshape(n, -1)
    feature_sets["H_ans_all"] = raw["h_ans"].reshape(n, -1)
    pca_dims = {"H_last_all": 64, "H_ans_all": 64}
    feature_sets["Msum_last_all"] = np.hstack(
        [feature_sets[f"Msum_last_l{l}"] for l in range(L)]
    )
    feature_sets["Msum_ans_all"] = np.hstack(
        [feature_sets[f"Msum_ans_l{l}"] for l in range(L)]
    )
    feature_sets["Msum_all"] = np.hstack(
        [feature_sets["Msum_last_all"], feature_sets["Msum_ans_all"]]
    )

    if args.raw_probes:
        raw_layers = sorted(set([0, L // 4, L // 2, 3 * L // 5, L - 1]))
        for l in raw_layers:
            feature_sets[f"Mraw_sel_ans_l{l}"] = raw["sel"][:, l, 1, :]
            feature_sets[f"Mraw_gate_ans_l{l}"] = raw["gate_pre"][:, l, 1, :]
            feature_sets[f"Mraw_down_ans_l{l}"] = raw["down"][:, l, 1, :]
            pca_dims[f"Mraw_sel_ans_l{l}"] = 256
            pca_dims[f"Mraw_gate_ans_l{l}"] = 256
            pca_dims[f"Mraw_down_ans_l{l}"] = 256

    # ---- run CV ----
    results = []
    oofs: dict[str, np.ndarray] = {}
    for name, X in feature_sets.items():
        print(f"  evaluating {name:>28} ({X.shape[1]} features) ...", flush=True)
        res = evaluate(X, y, groups, pca_dim=pca_dims.get(name))
        oof = res.pop("oof")
        oofs[name] = oof
        mask = ~np.isnan(oof)
        res["spearman_frac"] = float(
            spearmanr(oof[mask], frac[mask]).statistic
        )
        res.update({"feature_set": name})
        if name.startswith("H_last_l"):
            res["layer"] = int(name.rsplit("_l", 1)[1])
            res["family"] = "H_last"
        elif name.startswith("H_ans_l"):
            res["layer"] = int(name.rsplit("_l", 1)[1])
            res["family"] = "H_ans"
        elif name.startswith("Msum_last_l"):
            res["layer"] = int(name.rsplit("_l", 1)[1])
            res["family"] = "Msum_last"
        elif name.startswith("Msum_ans_l"):
            res["layer"] = int(name.rsplit("_l", 1)[1])
            res["family"] = "Msum_ans"
        elif name.startswith("Mraw"):
            res["layer"] = int(name.rsplit("_l", 1)[1])
            res["family"] = "Mraw"
        else:
            res["layer"] = -1
            res["family"] = name
        results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / f"{out_tag}_probe_results.csv", index=False)

    # nonlinear probes on the broad sets
    print("  MLP probes ...", flush=True)
    for name in ["H_ans_all", "Msum_all", "Msum_ans_all"]:
        if name in feature_sets:
            res = evaluate(feature_sets[name], y, groups, kind="mlp", pca_dim=pca_dims.get(name))
            oofs[f"mlp_{name}"] = res.pop("oof")
            res.update({"feature_set": f"mlp_{name}", "family": "mlp", "layer": -1})
            results.append(res)
    pd.DataFrame(results).to_csv(RESULTS_DIR / f"{out_tag}_probe_results.csv", index=False)

    # ---- prompt confidence AUROC comes from the 'confidence' row ----
    def best(family: str) -> pd.Series:
        sub = [r for r in results if r.get("family") == family]
        return max(sub, key=lambda r: r["auroc_oof"])

    print("\n=== headline (pooled OOF AUROC) ===")
    for fam in ["surface", "prob", "confidence", "confidence+surface", "H_last", "H_ans",
                "Msum_last", "Msum_ans", "Mraw", "mlp"]:
        sub = [r for r in results if r.get("family") == fam]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["auroc_oof"])
        print(
            f"  {fam:>20}: {b['auroc_oof']:.3f} "
            f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}] "
            f"mean {b['auroc_mean']:.3f}+-{b['auroc_std']:.3f} "
            f"({b['feature_set']})"
        )

    # ---- paired comparison + difficulty control + sanity checks ----
    extra_lines = []
    b_h = best("H_ans") if [r for r in results if r.get("family") == "H_ans"] else best("H_last")
    b_m = best("Msum_ans") if [r for r in results if r.get("family") == "Msum_ans"] else best("Msum_last")
    if b_h is not None and b_m is not None:
        d_mean, d_lo, d_hi = paired_delta(y, oofs[b_m["feature_set"]], oofs[b_h["feature_set"]])
        line = (
            f"- Paired bootstrap delta (MLP summaries - hidden states): "
            f"{d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
        )
        print(line)
        extra_lines.append(line)

    stage2_lines, _, _ = stage2_difficulty(
        df, y, df["frac_correct"].values, oofs, results, out_tag
    )
    print("\n".join(l for l in stage2_lines if l.startswith("|")))

    perm_lines = []
    rep_name = "H_ans_l" + str(L // 2)
    if rep_name in feature_sets:
        rng = np.random.default_rng(0)
        res_perm = evaluate(feature_sets[rep_name], rng.permutation(y), groups)
        perm_line = f"- Permutation check (`{rep_name}`, shuffled labels): AUROC {res_perm['auroc_oof']:.3f} (expected ~0.5)"
        print(perm_line)
        perm_lines.append(perm_line)

    make_figures(df, res_df, oofs, y, L, out_tag)
    make_report(df, res_df, oofs, y, out_tag, extra_lines=stage2_lines + extra_lines + perm_lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(df, res_df, oofs, y, L, tag):
    # 1. layer curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, prefix in zip(axes, ["H_last", "H_ans"]):
        sub = res_df[res_df["family"] == prefix]
        ax.plot(sub["layer"], sub["auroc_oof"], marker="o", ms=3)
        for f in ["Msum_last", "Msum_ans"]:
            sub2 = res_df[res_df["family"] == f]
            ax.plot(sub2["layer"], sub2["auroc_oof"], marker="s", ms=3, alpha=0.7)
        ax.set_title(prefix)
        ax.set_xlabel("layer")
        ax.axhline(0.5, ls="--", c="gray", lw=1)
    axes[0].set_ylabel("pooled OOF AUROC")
    fig.suptitle("Layer-wise probe AUROC (circles: hidden states, squares: MLP summaries)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_layer_curves.png", dpi=150)
    plt.close(fig)

    # 2. best per family
    fams = ["surface", "prob", "confidence", "confidence+surface", "H_last", "H_ans",
            "Msum_last", "Msum_ans", "Mraw", "mlp"]
    vals, names = [], []
    for f in fams:
        sub = [r for r in res_df.to_dict("records") if r["family"] == f]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["auroc_oof"])
        vals.append(b["auroc_oof"])
        names.append(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals, color="#4C72B0")
    ax.axvline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("pooled OOF AUROC")
    ax.set_title("Best probe per feature family")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_best_families.png", dpi=150)
    plt.close(fig)

    # 3. fraction-correct histogram + confidence relationship
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(df["frac_correct"], bins=31, color="#4C72B0")
    axes[0].set_title("fraction correct (K samples)")
    axes[0].set_xlabel("fraction correct")
    conf = np.clip(df["confidence"].values, 0, 100)
    axes[1].scatter(conf + np.random.default_rng(0).normal(0, 0.5, len(conf)),
                    df["frac_correct"], s=4, alpha=0.3)
    rho = spearmanr(conf, df["frac_correct"]).statistic
    axes[1].set_title(f"confidence vs frac_correct (rho={rho:.2f})")
    axes[1].set_xlabel("verbalized confidence")
    axes[2].scatter(df["p_gold"], df["frac_correct"], s=4, alpha=0.3, color="#DD8452")
    rho2 = spearmanr(df["p_gold"], df["frac_correct"]).statistic
    axes[2].set_title(f"p_gold vs frac_correct (rho={rho2:.2f})")
    axes[2].set_xlabel("model p(gold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_diagnostics.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def make_report(df, res_df, oofs, y, tag, extra_lines=None):
    recs = res_df.to_dict("records")

    def best(fam):
        sub = [r for r in recs if r["family"] == fam]
        return max(sub, key=lambda r: r["auroc_oof"]) if sub else None

    b_surf = best("surface")
    b_conf = best("confidence")
    b_prob = best("prob")
    b_h = best("H_ans") or best("H_last")
    b_m = best("Msum_ans") or best("Msum_last")
    b_raw = best("Mraw")
    b_mlp = best("mlp")

    lines = [f"# Pilot probe report (`{tag}`)", ""]
    lines.extend([
        "*This is an automatic run summary. See `../report/REPORT.md` and "
        "`../report/AUDIT.md` for audited definitions and interpretation. "
        "The feature pass uses a supplied gold token; metadata and probability "
        "baselines contain gold data. The summary chooses the answer-position "
        "MLP family before the last-prompt family. Best-variant selection is "
        "exploratory. The diagnostic gap does not remove all difficulty.*", "",
    ])
    lines.append(f"- questions: {len(df)}")
    lines.append(f"- positive rate (label): {y.mean():.3f}")
    lines.append(f"- greedy accuracy: {df['greedy_correct'].mean():.3f}")
    lines.append(f"- mean verbalized confidence: {np.clip(df['confidence'].values,0,100).mean():.1f}")
    lines.append("")
    lines.append("## Headline AUROCs (pooled out-of-fold)")
    lines.append("")
    lines.append("| feature family | best AUROC | 95% CI | best variant |")
    lines.append("| --- | --- | --- | --- |")
    for name, b in [
        ("surface (difficulty)", b_surf),
        ("model probs", b_prob),
        ("verbalized confidence", b_conf),
        ("hidden states", b_h),
        ("MLP summaries", b_m),
        ("MLP raw", b_raw),
        ("MLP nonlinear", b_mlp),
    ]:
        if b is None:
            continue
        lines.append(
            f"| {name} | {b['auroc_oof']:.3f} | "
            f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] | `{b['feature_set']}` |"
        )

    conf = np.clip(df["confidence"].values, 0, 100)
    lines.append("")
    lines.append("## Verbalized confidence vs difficulty")
    lines.append("")
    lines.append(
        f"- Spearman(confidence, log10 s_pop) = "
        f"{spearmanr(conf, np.log10(df['s_pop'].values + 1)).statistic:.3f}"
    )
    lines.append(
        f"- Spearman(confidence, frac_correct) = "
        f"{spearmanr(conf, df['frac_correct'].values).statistic:.3f}"
    )
    lines.append(
        f"- Spearman(confidence, p_gold) = "
        f"{spearmanr(conf, df['p_gold'].values).statistic:.3f}"
    )
    lines.append(
        f"- AUROC(confidence -> correct) = "
        f"{b_conf['auroc_oof']:.3f} (compared with probes above)"
    )

    lines.append("")
    lines.append("## Layer curves")
    lines.append("")
    lines.append(f"![layer curves](figures/{tag}_layer_curves.png)")
    lines.append(f"![best families](figures/{tag}_best_families.png)")
    lines.append(f"![diagnostics](figures/{tag}_diagnostics.png)")

    lines.append("")
    lines.append("## Verdict against hypotheses (auto-computed, to be interpreted)")
    lines.append("")
    if b_h and b_m:
        delta = b_m["auroc_oof"] - b_h["auroc_oof"]
        lines.append(
            f"- H1 (MLP signal > surface): MLP {b_m['auroc_oof']:.3f} vs surface "
            f"{b_surf['auroc_oof']:.3f} -> {'supported' if b_m['auroc_oof'] > b_surf['auroc_oof'] else 'not supported'}"
        )
        lines.append(
            f"- H2 (comparable to hidden): delta = {delta:+.3f} -> "
            f"{'supported' if abs(delta) <= 0.03 else 'not supported'}"
        )
    if b_mlp and b_h:
        lines.append(
            f"- Nonlinear MLP probe: {b_mlp['auroc_oof']:.3f} (family `{b_mlp['feature_set']}`)"
        )
    if b_raw:
        lines.append(
            f"- Raw MLP vector probe (exploratory): {b_raw['auroc_oof']:.3f} "
            f"(`{b_raw['feature_set']}`)"
        )
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    lines.append("")
    lines.append(
        "*Caveats: best layer/variant selected on pooled OOF scores (optimistic bias); "
        "single model, single dataset; K-sample labels are noisy; answer-alignment "
        "features use the gold token (oracle) and are not deployable as-is.*"
    )

    out = RESULTS_DIR / f"{tag}_report.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
