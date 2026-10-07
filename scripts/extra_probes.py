"""Targeted follow-up probes: combined hidden+MLP features (H3) and within-bin
difficulty checks for specific variants.

Saves OOF scores for reuse and prints a compact table.

Usage: python scripts/extra_probes.py --tag pilot7b [--label frac|greedy]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import RESULTS_DIR  # noqa: E402
from scripts.collect import SUMMARY_STATS  # noqa: E402
from scripts.probe_eval import (  # noqa: E402
    align,
    build_metadata_features,
    build_prob_features,
    evaluate,
    load_raw,
    paired_delta,
)

SEED = 0


def within_bin_table(df, y, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    pop = np.log10(df["s_pop"].values.astype(float) + 1.0)
    bins = pd.qcut(pop, 5, labels=False, duplicates="drop")
    rows = []
    for name, oof in scores.items():
        oof = np.asarray(oof, dtype=float)
        mask = np.isfinite(oof)
        row = {"score": name, "overall": roc_auc_score(y[mask], oof[mask])}
        for b in np.unique(bins):
            m = (bins == b) & mask
            row[f"bin{int(b)}"] = roc_auc_score(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="pilot7b")
    ap.add_argument("--label", choices=["frac", "greedy"], default="frac")
    args = ap.parse_args()

    feats = pd.read_parquet(RESULTS_DIR / f"{args.tag}_features.parquet")
    labels = pd.read_csv(RESULTS_DIR / f"{args.tag}_labels.csv")
    raw = load_raw(args.tag)
    df, raw = align(feats, labels, raw)
    frac = df["frac_correct"].values.astype(float)
    y = (frac >= 0.5).astype(int) if args.label == "frac" else df["greedy_correct"].values.astype(int)
    groups = df["subj_id"].values
    L = raw["h_last"].shape[1] - 1

    def msum(pos, l):
        return df[[f"m_{pos}_l{l}_{s}" for s in SUMMARY_STATS]].values.astype(np.float32)

    all_sum = np.hstack(
        [msum(p, l) for p in ("last", "ans") for l in range(L)]
    )
    h18 = raw["h_ans"][:, 18 + 1, :]  # post-layer-18 hidden state (index l+1)

    sets: dict[str, tuple[np.ndarray, int | None]] = {
        "H_ans_l18": (h18, None),
        "Msum_ans_l0": (msum("ans", 0), None),
        "Msum_last_l2": (msum("last", 2), None),
        "Mraw_sel_ans_l16": (raw["sel"][:, 16, 1, :], 256),
        "H18+Msum_ans_l18": (np.hstack([h18, msum("ans", 18)]), None),
        "H18+Msum_last_l2": (np.hstack([h18, msum("last", 2)]), None),
        "H18+Msum_ans_l0": (np.hstack([h18, msum("ans", 0)]), None),
        "H18+Msum_all": (np.hstack([h18, all_sum]), None),
        "surface": (build_metadata_features(df), None),
        "prob": (build_prob_features(df), None),
        "confidence": (np.clip(df["confidence"].values.astype(float), 0, 100).reshape(-1, 1), None),
    }

    oofs: dict[str, np.ndarray] = {}
    recs = []
    for name, (X, pca) in sets.items():
        print(f"evaluating {name} ({X.shape[1]} features) ...", flush=True)
        res = evaluate(X, y, groups, pca_dim=pca)
        oof = res.pop("oof")
        oofs[name] = oof
        mask = np.isfinite(oof)
        res.update(
            feature_set=name,
            spearman_frac=float(spearmanr(oof[mask], frac[mask]).statistic),
        )
        recs.append(res)

    out = pd.DataFrame(recs)[["feature_set", "auroc_oof", "ci_lo", "ci_hi", "n_features", "spearman_frac"]]
    out.to_csv(RESULTS_DIR / f"{args.tag}_{args.label}_extra.csv", index=False)
    np.savez(RESULTS_DIR / f"{args.tag}_{args.label}_extra_oof.npz", **oofs)

    print("\n=== extra probes ===")
    print(out.round(3).to_string(index=False))

    print("\n=== paired delta vs H_ans_l18 ===")
    for name in sets:
        if name == "H_ans_l18":
            continue
        d, lo, hi = paired_delta(y, oofs[name], oofs["H_ans_l18"])
        print(f"  {name:>22}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== within-bin AUROC (s_pop quintiles) ===")
    wb = within_bin_table(df, y, oofs)
    print(wb.round(3).to_string(index=False))
    wb.to_csv(RESULTS_DIR / f"{args.tag}_{args.label}_extra_withinbin.csv", index=False)

    print("\n=== spearman with surface residual (frac - surface_oof) ===")
    resid = frac - oofs["surface"]
    for name, oof in oofs.items():
        mask = np.isfinite(oof) & np.isfinite(resid)
        print(f"  {name:>22}: {spearmanr(oof[mask], resid[mask]).statistic:+.3f}")


if __name__ == "__main__":
    main()
