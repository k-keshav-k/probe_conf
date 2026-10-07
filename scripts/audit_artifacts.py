"""Check saved pilot evidence without generating answers or fitting new probes.

Run: .venv/bin/python -B scripts/audit_artifacts.py [--full-raw]
Writes only results/audit_checks.json. Historical result files are not changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import first_line, matches_any_alias, normalize_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-raw", action="store_true")
    args = ap.parse_args()
    res = ROOT / "results"
    qs = [json.loads(line) for line in (ROOT / "data/popqa_person.jsonl").read_text().splitlines()]
    lab = pd.read_csv(res / "pilot7b_labels.csv")
    feat = pd.read_parquet(res / "pilot7b_features.parquet")
    assert lab.id.is_unique and feat.id.is_unique
    assert len(qs) == len(lab) == len(feat) == 1087
    assert [q["id"] for q in qs] == lab.id.tolist() == feat.id.tolist()
    assert np.array_equal(lab.subj_id, feat.subj_id)
    assert np.array_equal(lab.prop, feat.prop)
    assert np.isfinite(feat.select_dtypes("number").to_numpy()).all()
    partial = pd.read_json(res / "pilot7b_features_partial.jsonl", lines=True)
    assert partial.id.tolist() == feat.id.tolist()
    for col in feat.select_dtypes("number"):
        assert np.allclose(partial[col], feat[col], rtol=1e-7, atol=1e-7), col

    samples = [json.loads(line) for line in (res / "pilot7b_samples.jsonl").read_text().splitlines()]
    assert [r["id"] for r in samples] == lab.id.tolist()
    counts = []
    strict_counts = []
    fallback_only = []
    three_char_aliases = 0
    for q, row in zip(qs, samples):
        assert row["question"] == q["question"] and len(row["samples"]) == 30
        flags = [matches_any_alias(first_line(s), q["aliases"]) for s in row["samples"]]
        counts.append(sum(flags))
        strict_count = 0
        three_char_aliases += sum(len(normalize_answer(a)) == 3 for a in q["aliases"])
        for s, ok in zip(row["samples"], flags):
            tokens = normalize_answer(first_line(s)).split()
            strict = any(
                (tokens == normalize_answer(a).split()) if len(normalize_answer(a)) <= 2
                else any(tokens[i:i + len(normalize_answer(a).split())] == normalize_answer(a).split()
                         for i in range(len(tokens) - len(normalize_answer(a).split()) + 1))
                for a in q["aliases"] if normalize_answer(a)
            )
            if ok and not strict:
                fallback_only.append({"id": q["id"], "text": first_line(s)})
            strict_count += int(strict)
        strict_counts.append(strict_count)
    assert np.allclose(np.asarray(counts) / 30, lab.frac_correct, atol=1e-7)
    frac = lab.frac_correct.to_numpy()
    yf = (frac >= 0.5).astype(int)
    yg = lab.greedy_correct.to_numpy().astype(int)
    conf = lab.confidence.to_numpy()
    popbins = pd.qcut(np.log10(feat.s_pop.to_numpy() + 1), 5, labels=False)

    output = {
        "questions": len(lab), "unique_subjects": int(lab.subj_id.nunique()),
        "relation_counts": dict(Counter(lab.prop)), "feature_shape": list(feat.shape),
        "sample_count": len(samples) * 30, "sample_label_replay": "pass",
        "mean_fraction": float(frac.mean()), "fraction_positive": int(yf.sum()),
        "greedy_positive": int(yg.sum()), "confidence_mean": float(conf.mean()),
        "confidence_median": float(np.median(conf)),
        "invalid_confidence": int(((conf < 1) | (conf > 100)).sum()),
        "confidence_fraction_spearman": float(spearmanr(conf, frac).statistic),
        "confidence_popularity_spearman": float(spearmanr(conf, feat.s_pop).statistic),
        "raw_confidence_auroc": float(roc_auc_score(yf, conf)),
        "p_gold_auroc": float(roc_auc_score(yf, feat.p_gold)),
        "substring_only_matches": fallback_only,
        "strict_match_diagnostic": {
            "sample_matches": int(sum(strict_counts)),
            "mean_fraction": float(np.mean(strict_counts) / 30),
            "positive_count": int(sum(c >= 15 for c in strict_counts)),
            "threshold_changes": [
                {"id": q["id"], "question": q["question"],
                 "historical_fraction": float(f), "token_boundary_fraction": c / 30}
                for q, f, c in zip(qs, frac, strict_counts) if (f >= 0.5) != (c >= 15)
            ],
        },
        "three_character_alias_count": three_char_aliases,
        "popularity_bins": [], "confidence_bins": [], "scores": {}, "paired": {},
    }
    for b in range(5):
        m = popbins == b
        output["popularity_bins"].append({
            "bin": b, "n": int(m.sum()), "min_views": int(feat.s_pop[m].min()),
            "max_views": int(feat.s_pop[m].max()), "median_views": float(feat.s_pop[m].median()),
            "fraction_positive_rate": float(yf[m].mean()), "mean_fraction": float(frac[m].mean()),
            "greedy_accuracy": float(yg[m].mean()), "confidence_mean": float(conf[m].mean()),
        })
    for lo, hi in zip([0, 60, 80, 90, 95], [60, 80, 90, 95, 100]):
        m = (conf > lo) & (conf <= hi)
        output["confidence_bins"].append({"lo_exclusive": lo, "hi_inclusive": hi,
            "n": int(m.sum()), "fraction_positive_rate": float(yf[m].mean()),
            "mean_fraction": float(frac[m].mean()), "greedy_accuracy": float(yg[m].mean())})
    for target, y in [("frac", yf), ("greedy", yg)]:
        folds = []
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(feat, y, lab.subj_id):
            assert not set(lab.subj_id.iloc[tr]) & set(lab.subj_id.iloc[te])
            folds.append({"train": len(tr), "test": len(te), "test_positive": int(y[te].sum())})
        output[f"{target}_folds"] = folds
        table = pd.read_csv(res / f"pilot7b_{target}_extra.csv").set_index("feature_set")
        within = pd.read_csv(res / f"pilot7b_{target}_extra_withinbin.csv").set_index("score")
        with np.load(res / f"pilot7b_{target}_extra_oof.npz") as z:
            scores = {k: z[k].copy() for k in z.files}
        output["scores"][target] = {}
        for name, score in scores.items():
            assert score.shape == (len(y),) and np.isfinite(score).all()
            auc = float(roc_auc_score(y, score))
            assert np.isclose(auc, table.loc[name, "auroc_oof"], atol=1e-12)
            assert np.isclose(auc, within.loc[name, "overall"], atol=1e-12)
            for b in range(5):
                m = popbins == b
                assert np.isclose(roc_auc_score(y[m], score[m]), within.loc[name, f"bin{b}"], atol=1e-12)
            rho = float(spearmanr(score, frac - scores["surface"]).statistic)
            output["scores"][target][name] = {"auroc": auc, "surface_gap_spearman": rho}
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(1000):
                idx = rng.choice(len(y), size=len(y), replace=True)
                if len(np.unique(y[idx])) == 2:
                    boot.append(roc_auc_score(y[idx], score[idx]))
            assert np.allclose(np.percentile(boot, [2.5, 97.5]), table.loc[name, ["ci_lo", "ci_hi"]].to_numpy(float), atol=1e-12)
        pairs = [("H18+Msum_all", "H_ans_l18"), ("Msum_last_l2", "H_ans_l18"), ("H18+Msum_all", "prob")]
        output["paired"][target] = {}
        for a, b in pairs:
            rng = np.random.default_rng(0)
            ds = []
            for _ in range(2000):
                idx = rng.choice(len(y), size=len(y), replace=True)
                if len(np.unique(y[idx])) == 2:
                    ds.append(roc_auc_score(y[idx], scores[a][idx]) - roc_auc_score(y[idx], scores[b][idx]))
            output["paired"][target][f"{a} minus {b}"] = {
                "point": float(roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b])),
                "bootstrap_mean": float(np.mean(ds)), "ci": np.percentile(ds, [2.5, 97.5]).tolist()}

    files = sorted((res / "raw").glob("pilot7b_batch*.npz"))
    ids = []
    checked_summaries = 0
    for f in files:
        with np.load(f) as z:
            batch_ids = z["ids"]
            ids.extend(batch_ids.tolist())
            B = len(batch_ids)
            expected = {"h_last": (29, B, 3584), "h_ans": (29, B, 3584),
                        "gate_pre": (B, 28, 2, 18944), "sel": (B, 28, 2, 18944), "down": (B, 28, 2, 3584)}
            with zipfile.ZipFile(f) as archive:
                for key, shape in expected.items():
                    with archive.open(f"{key}.npy") as entry:
                        version = np.lib.format.read_magic(entry)
                        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        actual, _, dtype = reader(entry)
                    assert actual == shape and dtype == np.dtype("float16"), (f, key, actual)
                    if args.full_raw:
                        assert np.isfinite(z[key]).all(), (f, key)
            if args.full_raw:
                from scripts.collect import entropy_logits_np, gini_np, topk_mass_np, EPS
                gate = z["gate_pre"].astype(np.float32)
                sel = z["sel"].astype(np.float32)
                down = z["down"].astype(np.float32)
                hs = np.stack([z["h_last"].astype(np.float32), z["h_ans"].astype(np.float32)], axis=2)
                hs = hs.transpose(1, 0, 2, 3)[:, 1:]
                mu = gate.mean(-1, keepdims=True)
                sd = gate.std(-1, keepdims=True) + EPS
                dn = np.linalg.norm(down, axis=-1)
                hn = np.linalg.norm(hs, axis=-1)
                values = {
                    "ent_gate": entropy_logits_np(gate),
                    "ent_gate_sq": entropy_logits_np(np.log(gate * gate + EPS)),
                    "active_frac": (gate > 0).mean(-1),
                    "top5_gate": topk_mass_np(gate * gate, 5),
                    "gini_gate": gini_np(gate), "max_gate": gate.max(-1),
                    "kurt_gate": ((gate-mu)**4).mean(-1) / (sd[...,0]**4),
                    "ent_sel_sq": entropy_logits_np(np.log(sel * sel + EPS)),
                    "l2_sel": np.linalg.norm(sel, axis=-1), "l2_down": dn,
                    "down_norm_ratio": dn / (hn + EPS),
                    "cos_down_h": (down * hs).sum(-1) / (dn * (hn + EPS)),
                }
                rows = feat.set_index("id").loc[batch_ids]
                for name, v in values.items():
                    for pos, label in enumerate(("last", "ans")):
                        cols = [f"m_{label}_l{l}_{name}" for l in range(28)]
                        assert np.allclose(rows[cols].to_numpy(), v[:,:,pos], rtol=1e-6, atol=1e-6), (f,name,label)
                        checked_summaries += B * 28
    assert ids == lab.id.tolist()
    output["raw"] = {"batches": len(files), "ids_and_shapes": "pass",
                     "all_values_finite": "pass" if args.full_raw else "not checked",
                     "nonalignment_summary_values_recomputed": checked_summaries}
    output["verified"] = ["question/label/feature/raw ID order", "feature JSONL/Parquet agreement",
        "30-sample labels replayed", "OOF AUROCs", "OOF confidence intervals", "within-bin AUROCs",
        "subject separation in recreated folds", "raw tensor headers"]
    (res / "audit_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
