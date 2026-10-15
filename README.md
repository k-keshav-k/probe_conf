# probe_conf — confidence, difficulty, and MLP-interaction probes

Pilot implementation of the preliminary experiment in `PLAN.md`: train probes on
hidden states vs. MLP-interaction features to predict whether Qwen2.5 answers
PopQA person questions correctly, with an eye on difficulty (popularity) control.

Evidence review: 3 October 2026. Saved headline scores reproduce. The report
now specifies gold dependencies, matcher limits, and evaluation limits.
Start with `report/REPORT.html` and `report/AUDIT.md`.

## Files

- `meet.txt` / `meet.md` — formatted meeting notes (original preserved in `meet.original.txt`)
- [Causal confidence training plan](report/CAUSAL_CONFIDENCE_TRAINING_PLAN.md) — proposed follow-up to the selective-metacognition toy: objectives, staged experiments, controls, and decision gates
- `causal_confidence/` — implementation of that plan: frozen protocol, subject-disjoint data splits, retained-state episodes, channel interventions, immutable outcome banks, scalar and native reporters, exact/pair objectives, baselines, clustered evaluation, and the P0/P1/P2 runners. P3-P6 are gated. See `causal_confidence/cluster/README.md` for cluster submission.
- `LITERATURE.md` — literature notes: geometric recall, ROME lineage, confidence
  vs. difficulty, internal-state probes
- `PLAN.md` — experiment plan, hypotheses, feature design, decision rules
- `EXPERIMENTS.md` — chronological run log
- `report/REPORT.md`, `report/REPORT.html`, `report/REPORT.pdf` — paper-style
  report using ASD-STE100 writing principles, with figures, citations,
  and appendices (the HTML is a single self-contained file with embedded images)
- `report/EXPLAINER.html` — interactive explainer of the internal signals
  (layers, MLP steps, the 17 numbers, raw vectors). Single self-contained file.
- `report/FLOW.html` — three detailed panels: collection, internal computation
  and hooks, and probe training and evaluation. Includes an enlargement control.
- `report/AUDIT.md`, `results/audit_checks.json` — findings and numerical checks.
- `results/FINDINGS.md` — technical pilot results and interpretation
- `section33/README.md` — how to run the confirmatory experiment
- `cluster/README.md` — NYU Torch runbook (Slurm, storage, Apptainer overlay,
  prefetch, job scripts)
- `src/` — pipeline code
- `scripts/` — entry points
- `data/`, `results/` — generated artifacts

## Quick start

```bash
bash scripts/setup_env.sh                     # ROCm gfx1151 torch + HF stack
.venv/bin/python scripts/smoke_test.py        # GPU + model + hooks sanity check
.venv/bin/python scripts/prepare_data.py --force --n 1200 --mix head
.venv/bin/python scripts/collect.py --model Qwen/Qwen2.5-7B-Instruct \
    --k 30 --batch-questions 8 --tag pilot7b --fewshot
.venv/bin/python scripts/probe_eval.py --tag pilot7b --raw-probes
.venv/bin/python scripts/probe_eval.py --tag pilot7b --label greedy --raw-probes
.venv/bin/python scripts/extra_probes.py --tag pilot7b --label frac
.venv/bin/python scripts/extra_probes.py --tag pilot7b --label greedy
# or everything: bash scripts/run_pilot.sh
```

Hardware note: the machine is an AMD Strix Halo (Radeon 8060S, gfx1151). The
standard PyTorch ROCm 6.4 wheels do **not** contain gfx1151 kernels; `setup_env.sh`
uses AMD's gfx1151 wheel index (`torch 2.10.0+rocm7.13.0a`, 15.6 TFLOPS fp16,
57 GB usable).

Collection commands overwrite named run artifacts. Use a new tag for a new
experiment. To inspect and present saved evidence without model generation:

```bash
.venv/bin/python -B scripts/audit_artifacts.py --full-raw
PYTHONPATH=. .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/make_flow_diagrams.py
.venv/bin/python -B scripts/make_report_figures.py
.venv/bin/python -B scripts/make_explainer.py
node tests/check_explainer.js
.venv/bin/python -B scripts/make_report_html.py
.venv/bin/python -B scripts/make_report_pdf.py
```

## Causal confidence package

```bash
.venv/bin/python -m causal_confidence.smoke          # fast contract checks on CPU
.venv/bin/python -m causal_confidence.smoke --end-to-end   # + Qwen2.5-0.5B P1/P2 mechanics
.venv/bin/python -m pytest tests/test_causal_confidence.py -q
```

Stage commands for the 7B runs (`setup`, `p0`, `p1`, `p2-discover`,
`p2-validate`, `audit`) are documented in
`causal_confidence/cluster/README.md`. P3-P6 refuse to run before the P2
gate passes.
