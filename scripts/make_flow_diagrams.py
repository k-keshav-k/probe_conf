"""Draw the implemented pipeline; save PNG, SVG, and a standalone HTML page."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/figures"
COL = {"model": "#eaf1fb", "data": "#eef2f5", "gold": "#fff0d7",
       "read": "#e8f5ef", "train": "#f1eafb", "warn": "#fff0ee"}


class Panel:
    def __init__(self, title, subtitle, height=940):
        self.fig, self.ax = plt.subplots(figsize=(8.8, height / 112))
        self.ax.set(xlim=(0, 1000), ylim=(height, 0))
        self.ax.axis("off")
        self.height = height
        self.boxes = {}
        self.ax.text(18, 25, title, fontsize=14, weight="bold", va="top", color="#182b40")
        self.ax.text(18, 63, subtitle, fontsize=10, va="top", color="#45596b")

    def box(self, key, x, y, w, h, title, lines, kind="data", fs=10):
        self.boxes[key] = (x, y, w, h)
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                      fc=COL[kind], ec="#718397", lw=0.9))
        self.ax.text(x + 12, y + 12, title, va="top", fontsize=fs, weight="bold", color="#182b40")
        self.ax.text(x + 12, y + 37, "\n".join(lines), va="top", fontsize=fs - 0.4,
                     linespacing=1.35, color="#213243")

    def edge(self, start, end, a="bottom", b="top", color="#465b70", dash=False):
        def port(key, side):
            x, y, w, h = self.boxes[key]
            return {"top": (x+w/2,y), "bottom": (x+w/2,y+h),
                    "left": (x,y+h/2), "right": (x+w,y+h/2)}[side]
        self.ax.add_patch(FancyArrowPatch(port(start,a),port(end,b),arrowstyle="-|>",
                          mutation_scale=12,lw=1.3,color=color,linestyle="--" if dash else "-"))

    def note(self, y, text):
        self.ax.text(18, y, text, va="top", fontsize=9.6, color="#45596b", linespacing=1.4)

    def path(self, points, dash=False, color="#465b70"):
        self.ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                     color=color, lw=1.3, ls="--" if dash else "-")
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                         mutation_scale=12, lw=1.3, color=color, ls="--" if dash else "-"))

    def save(self, name):
        self.fig.subplots_adjust(left=0.015, right=0.995, top=0.995, bottom=0.015)
        self.fig.savefig(OUT / f"{name}.png", dpi=210, facecolor="white")
        with plt.rc_context({"svg.fonttype": "none"}):
            self.fig.savefig(OUT / f"{name}.svg", facecolor="white")
        plt.close(self.fig)


def collection():
    p = Panel("A. From questions to labels and recorded signals",
              "Three separate model-call paths. Language-model weights stay fixed.", 970)
    p.box("data", 150, 104, 700, 100, "1. Prepare the selected question set",
          ["PopQA → four relations → weighted popularity sample",
           "1,087 questions; 1,046 subject IDs; gold object and aliases"], "data")
    p.box("answer", 18, 252, 300, 136, "2a. Generate answers",
          ["System + four examples + question",
           "30 samples: T=1, top_p=1",
           "One greedy output; ≤16 new tokens"], "model")
    p.box("conf", 349, 252, 300, 136, "2b. Request confidence",
          ["System + question + number request",
           "No four examples; no supplied answer",
           "Greedy output; ≤8 new tokens"], "model")
    p.box("pass", 680, 252, 300, 136, "2c. Run the feature pass",
          ["Answer prompt p + one gold token g",
           "g = first token of space + gold object",
           "Read last=len(p)−1 and ans=len(p)"], "gold")
    for key in ("answer", "conf", "pass"):
        p.edge("data", key)
    p.box("match", 18, 435, 300, 145, "3a. Compare with gold aliases",
          ["First nonempty line; normalize text",
           "Alias token match or substring match",
           "f = accepted samples / 30",
           "y_frac = [f ≥ 0.5]; greedy label = 0 or 1"], "gold", fs=9.4)
    p.box("parse", 349, 435, 300, 145, "3b. Parse confidence",
          ["First integer from 1 to 100",
           "Failed parse → −1 (three questions)",
           "Historical probe clips failure to 0",
           "No specific answer is supplied"], "data")
    p.box("read", 680, 435, 300, 145, "3c. Read internal signals",
          ["28 layers × two token positions",
           "Hooks: gate_pre, sel, down",
           "Model output: hidden states + logits",
           "Panel B expands this path"], "read")
    p.edge("answer", "match"); p.edge("conf", "parse"); p.edge("pass", "read")
    p.box("labels", 18, 630, 300, 123, "4a. Save labels and samples",
          ["CSV labels: f, greedy, confidence",
           "JSONL samples: 30 outputs",
           "Gold aliases define correctness labels"], "data", fs=9.7)
    p.box("feature", 515, 630, 465, 123, "4b. Save feature rows and raw tensors",
          ["Parquet / JSONL: 952 summaries + other features",
           "NPZ: 29 hidden states and all 28 MLP layers",
           "Saved float16 activations; question IDs join the records"], "read", fs=9.7)
    p.edge("match", "labels"); p.edge("parse", "labels"); p.edge("read", "feature")
    p.box("train", 170, 802, 660, 103, "5. Train and test auxiliary probes — see Panel C",
          ["Align by question ID; split by subject ID; train only the probe",
           "Score held-out questions; compute AUROC and diagnostic tables"], "train")
    p.edge("labels", "train"); p.edge("feature", "train")
    p.note(928, "Gold has two roles: it defines labels and supplies oracle feature information.\nThe model-call paths do not read probe scores or receive probe-training updates.")
    p.save("flow_a_collection")


def internals():
    p = Panel("B. Inside one transformer layer: computation, reads, and summaries",
              "The same operations run at both positions and all 28 layers. A hook copies data.", 1110)
    p.box("h", 18, 105, 390, 84, "1. Layer input h — 3,584 values",
          ["Token embedding at layer 0; prior state later"], "model")
    p.box("att", 18, 225, 390, 102, "2. Normalize, apply attention, and add",
          ["Attention reads permitted earlier token states",
           "r = h + Attention(RMSNorm(h))"], "model")
    p.edge("h", "att")
    p.box("norm", 18, 365, 390, 84, "3. Form the MLP input",
          ["x = RMSNorm(r); x has 3,584 values"], "model")
    p.edge("att", "norm")
    p.box("gate", 18, 489, 210, 99, "4a. Gate projection",
          ["g_pre = W_gate x",
           "18,944 signed values"], "model",fs=9.8)
    p.box("up", 258, 489, 210, 99, "4b. Up projection",
          ["u = W_up x",
           "18,944 signed values"], "model",fs=9.8)
    p.edge("norm", "gate"); p.edge("norm", "up")
    p.box("sel", 18, 629, 450, 100, "5. Apply SiLU and multiply elementwise",
          ["a = SiLU(g_pre); s = a ⊙ u",
           "s has 18,944 values; gates are not binary"], "model")
    p.edge("gate", "sel"); p.edge("up", "sel")
    p.box("down", 18, 770, 450, 83, "6. Project to residual width",
          ["w = W_down s; w has 3,584 values"], "model")
    p.edge("sel", "down")
    p.box("sum", 18, 890, 450, 107, "7. Add the MLP output",
          ["h_next = r + w → next transformer layer",
           "After layer 27: final RMSNorm → output head"], "model")
    p.edge("down", "sum")
    p.box("pos", 535, 105, 445, 130, "Read positions and causal information",
          ["last: last prompt token; cannot read later g",
           "ans: supplied gold token; can read prompt + g",
           "All ans activations have gold-input dependency"], "gold")
    p.box("hooks", 535, 288, 445, 144, "Hook copies → CPU float16 → raw NPZ",
          ["gate_proj output → gate_pre",
           "down_proj input → sel",
           "down_proj output → down",
           "Separate hidden-state output → h_last / h_ans"], "read")
    p.path([(228,538),(241,538),(241,606),(495,606),(495,326),(535,326)], dash=True)
    p.path([(468,678),(508,678),(508,370),(535,370)], dash=True)
    p.path([(468,810),(520,810),(520,405),(535,405)], dash=True)
    p.path([(18,278),(7,278),(7,944),(18,944)], color="#465b70")
    p.box("summary", 535, 482, 445, 160, "Compute summaries from saved copies",
          ["Reconstruct float32 arrays after copy quantization",
           "12 distribution / norm / cosine features",
           "5 gold-direction / candidate features",
           "17 × 28 layers × 2 positions = 952 values"], "read")
    p.edge("hooks", "summary")
    p.box("gold", 535, 698, 445, 169, "Gold directions and candidate scores",
          ["D_v: output-head row; A_v = D_v W_down",
           "last final logits → top 50 tokens + gold g",
           "Local score z_v = A_v · s",
           "Gold alignment, local margin, candidate entropy",
           "These are not complete-answer probabilities"], "gold")
    p.edge("gold", "summary", "top", "bottom", color="#ab7025")
    p.box("bank", 535, 921, 445, 105, "Stored inputs to separate auxiliary probes",
          ["Hidden vectors, scalar summaries, raw vectors",
           "Raw storage: all layers; raw probe sweep: five"], "read")
    p.path([(980,562),(993,562),(993,973),(980,973)], dash=True)
    p.edge("sum", "bank", "right", "left", dash=True)
    p.note(1056, "Solid arrows: computation. Dashed arrows: read or storage path. Only probe parameters train.\nHidden index l+1 gives the layer-l read; the final hidden read includes final RMSNorm.")
    p.save("flow_b_signals")


def training():
    p = Panel("C. From recorded signals to trained probes and held-out scores",
              "The reported training uses fixed language-model activations and historical matcher labels.", 1110)
    p.box("input", 125, 103, 750, 129, "1. Align three inputs by question ID",
          ["X: recorded features; y: main or greedy label; group: subject ID",
           "Examples: hidden=3,584; MLP=17; all summaries=952",
           "Combined X = hidden layer 18 at ans + 952 summaries = 4,536"], "data")
    p.box("split", 125, 270, 750, 97, "2. Make five subject-disjoint outer folds",
          ["StratifiedGroupKFold; seed 0; one subject cannot cross train / test",
           "Repeat steps 3–6 for each held-out fold"], "train")
    p.edge("input", "split")
    p.box("train", 18, 412, 570, 133, "3. Fit preprocessing on outer training features",
          ["StandardScaler: training means and standard deviations",
           "Optional PCA: raw vectors → 256; all hidden → 64",
           "No PCA in the headline hidden, MLP, or combined sets"], "train")
    p.box("test", 635, 412, 345, 133, "Outer test questions",
          ["Keep features and labels held out",
           "Their subjects are absent from train",
           "Transform with training scaler / PCA"], "data")
    p.edge("split", "train"); p.edge("split", "test")
    p.box("inner", 18, 583, 570, 126, "4. Select logistic-regression regularization C",
          ["Three inner subject-disjoint folds; mean validation AUROC",
           "C ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}",
           "Saved run reuses outer-fitted preprocessing inside inner folds"], "train", fs=9.8)
    p.edge("train", "inner")
    p.box("fit", 18, 748, 570, 105, "5. Refit probe on all outer training rows",
          ["Training X and y learn coefficients β and intercept b",
           "L2 logistic loss; language-model weights remain fixed"], "train")
    p.edge("inner", "fit")
    p.box("score", 635, 748, 345, 105, "6. Score outer test rows",
          ["score = sigmoid(b + β · X_test)",
           "Store one out-of-fold score per row"], "read", fs=9.7)
    p.edge("fit", "score", "right", "left")
    p.edge("test", "score")
    p.box("pool", 125, 893, 750, 112, "7. Pool scores from all five folds → report metrics",
          ["AUROC; 1,000 question-bootstrap samples; 2,000 paired samples",
           "Same OOF scores within popularity bins; no bin-specific retraining",
           "Diagnostic gap = sample fraction − binary metadata-probe score"], "read", fs=9.8)
    p.edge("score", "pool")
    p.note(1035, "Gold dependencies remain in X. Best-layer selection and subject clustering are not in the intervals.\nNo final deployed classifier or rejection threshold is saved. A live-use procedure needs a new test.")
    p.save("flow_c_training")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collection(); internals(); training()
    names = ["flow_a_collection", "flow_b_signals", "flow_c_training"]
    titles = ["A. Collection and labels", "B. Internal computation and reads", "C. Probe training and evaluation"]
    content = "\n".join(f'<section id="panel-{i}"><h2>{title}</h2><svg{(OUT / (name + ".svg")).read_text().split("<svg",1)[1]}</section>'
                        for i, (name,title) in enumerate(zip(names,titles)))
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complete flow: internal signals and auxiliary probes</title>
<style>body{margin:0;background:#f2f5f8;color:#172c41;font:17px/1.6 system-ui}
header,main,footer{max-width:1160px;margin:auto;padding:24px}h1{font-size:28px}
nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#1d5289}section{background:white;padding:18px;margin:24px 0;border-radius:12px}
svg{display:block;width:100%;height:auto}button{font:inherit;padding:7px 14px;margin-right:8px}
.scroll{overflow:auto}main.enlarged{max-width:none}main.enlarged section{min-width:1500px;max-width:1700px;margin:24px auto}
@media print{header button{display:none}body{background:white}section{break-before:page;padding:0}header,main,footer{padding:0}}</style>
<header><h1>Complete flow of the offline probe experiment</h1>
<p>Read the three panels in order. They show the implemented calls, signal reads, feature calculations, and probe training.</p>
<nav><a href="#panel-0">Collection</a><a href="#panel-1">Internal signals</a><a href="#panel-2">Training</a><a href="REPORT.html">Full report</a><a href="EXPLAINER.html">Recorded-value explorer</a></nav>
<p><button id="zoom">Enlarge diagrams</button><button onclick="window.print()">Print</button></p>
<p>Gold data define the labels and also enter some feature inputs. Only auxiliary probe parameters train.</p></header>
<div class="scroll"><main>''' + content + '''</main></div>
<footer><p>Source: scripts/collect.py, scripts/probe_eval.py, and scripts/extra_probes.py. Evidence review: 3 October 2026.</p>
<p>The proposed cross-layer attention model and a live rejection policy are not implemented.</p></footer>
<script>document.getElementById('zoom').onclick=function(){let x=document.querySelector('main').classList.toggle('enlarged');this.textContent=x?'Fit diagrams to page':'Enlarge diagrams';};</script></html>'''
    (ROOT / "report/FLOW.html").write_text(html)
    print("Wrote three PNG / SVG panels and report/FLOW.html")


if __name__ == "__main__":
    main()
