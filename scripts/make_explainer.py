"""Build the interactive HTML explainer for the internal-signal section.

Output: report/EXPLAINER.html (single file, no external resources).

The page explains, with real pilot data:
  1. the 28 transformer layers and the two signals recorded at each layer,
  2. the four steps inside one MLP (keys -> nonlinearity -> values -> write),
  3. the 17 summary numbers (grouped: gates, write, alignment, competition),
  4. the raw MLP vectors probed at five layers (saved at all layers).

Usage: .venv/bin/python scripts/make_explainer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "EXPLAINER.html"

STATS = [
    "ent_gate", "ent_gate_sq", "active_frac", "top5_gate", "gini_gate",
    "max_gate", "kurt_gate", "ent_sel_sq", "l2_sel", "l2_down",
    "down_norm_ratio", "cos_down_h", "cos_sel_agold", "proj_gold_norm",
    "mlp_margin", "ent_cand", "logit_mlp_gold",
]

EXAMPLES = {"correct": 1762842, "wrong": 1657474}  # Donald Trump / Wayne Gretzky

N_LAYERS = 28
BATCH = 8


# ---------------------------------------------------------------------------
def load_series() -> dict:
    probe = pd.read_csv(RES / "pilot7b_probe_results.csv")
    out = {}
    for key, family in [
        ("hidden_ans", "H_ans"), ("mlp_ans", "Msum_ans"),
        ("hidden_last", "H_last"), ("mlp_last", "Msum_last"),
    ]:
        sub = probe[probe.family == family].sort_values("layer")
        out[key] = [round(float(v), 4) for v in sub["auroc_oof"]]
    out["layers"] = list(range(N_LAYERS))
    return out


def load_example_stats(feat: pd.DataFrame) -> dict:
    out = {}
    for key, qid in EXAMPLES.items():
        entry = {"stats": {"ans": {}, "last": {}}}
        for pos in ("ans", "last"):
            for s in STATS:
                vals = [
                    float(feat.loc[qid, f"m_{pos}_l{l}_{s}"]) for l in range(N_LAYERS)
                ]
                entry["stats"][pos][s] = [round(v, 5) for v in vals]
        out[key] = entry
    return out


def load_question_meta(labels: pd.DataFrame) -> dict:
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "popqa_person.jsonl")}
    out = {}
    for key, qid in EXAMPLES.items():
        row = labels[labels.id == qid].iloc[0]
        out[key] = {
            "id": int(qid),
            "question": qs[qid]["question"],
            "subject": qs[qid]["subj"],
            "gold": qs[qid]["obj"],
            "confidence": float(row["confidence"]),
            "frac": float(row["frac_correct"]),
            "greedy": int(row["greedy_correct"]),
            "prop": qs[qid]["prop"],
        }
    return out


def load_gate_data() -> dict:
    """Top-20 key matches and a 48-bin histogram for each example and layer."""
    questions = [json.loads(l) for l in open(ROOT / "data" / "popqa_person.jsonl")]
    order = {q["id"]: i for i, q in enumerate(questions)}
    out = {}
    for key, qid in EXAMPLES.items():
        i = order[qid]
        batch_start = (i // BATCH) * BATCH
        z = np.load(RES / "raw" / f"pilot7b_batch{batch_start:05d}.npz")
        row = int(np.where(z["ids"] == qid)[0][0])
        gate = z["gate_pre"][row].astype(np.float32)  # (28, 2, 18944)
        out[key] = {}
        for p, pos in enumerate(("last", "ans")):
            top, hist = [], []
            for l in range(N_LAYERS):
                vals = gate[l, p, :]
                srt = np.sort(vals)[::-1]
                top.append([round(float(v), 3) for v in srt[:20]])
                counts, edges = np.histogram(vals, bins=48)
                hist.append({
                    "counts": [int(c) for c in counts],
                    "min": round(float(edges[0]), 3),
                    "max": round(float(edges[-1]), 3),
                })
            out[key][pos] = {"top": top, "hist": hist}
    return out


def main() -> None:
    feat = pd.read_parquet(RES / "pilot7b_features.parquet").set_index("id")
    labels = pd.read_csv(RES / "pilot7b_labels.csv")
    extra = pd.read_csv(RES / "pilot7b_frac_extra.csv").set_index("feature_set")

    data = {
        "series": load_series(),
        "example_stats": load_example_stats(feat),
        "example_meta": load_question_meta(labels),
        "gate": load_gate_data(),
        "raw_layers": sorted({0, N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 5, N_LAYERS - 1}),
        "d_ff": 18944,
        "stat_names": STATS,
        "headline": {
            "raw": round(float(extra.loc["Mraw_sel_ans_l16", "auroc_oof"]), 3),
            "mlp": round(float(extra.loc["Msum_last_l2", "auroc_oof"]), 3),
            "hidden": round(float(extra.loc["H_ans_l18", "auroc_oof"]), 3),
            "combined": round(float(extra.loc["H18+Msum_all", "auroc_oof"]), 3),
        },
    }

    template = (Path(__file__).resolve().parent / "explainer_template.html").read_text()
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} kB)")
    print("raw layers:", data["raw_layers"])
    for k, v in data["example_meta"].items():
        print(k, v["question"], "->", v["gold"], "| frac", v["frac"], "| conf", v["confidence"])


if __name__ == "__main__":
    main()
