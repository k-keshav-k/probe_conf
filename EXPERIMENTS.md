# Experiment log — probe_conf pilot

Chronological record of what was run, on what hardware, with what outcome.
The pilot is complete. Evidence review: 2026-10-03. `report/REPORT.md` and
`report/AUDIT.md` supersede earlier interpretations below. Historical scores
remain unchanged. Gold dependencies, matcher issues, and inference limits apply.

## 2026-09-28 — environment

- Machine: AMD Ryzen AI Max (Strix Halo), Radeon 8060S iGPU, gfx1151, 59 GB RAM,
  57 GB usable GPU memory, Ubuntu, kernel 7.0.0-34.
- Python 3.14 system + `uv`; created `.venv` with Python 3.12.
- **GPU software problem:** `torch 2.9.1+rocm6.4` (official ROCm wheel) has **no
  gfx1151 kernels** → `HIP error: invalid device function` on any op.
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` → `no kernel image is available`.
- **Fix:** AMD gfx1151 wheel index (`https://rocm.nightlies.amd.com/v2/gfx1151/`):
  `torch 2.10.0+rocm7.13.0a20260513`, `rocm-sdk-libraries-gfx1151`.
  Verified: 15.6 TFLOPS fp16 matmul (4096³), 8 GB allocation, SDPA forward OK.
  Stable alternative documented in `scripts/setup_env.sh` (AMD
  `whl-multi-arch` index, torch 2.12.0+rocm7.14.1).

## Smoke tests (Qwen2.5-0.5B-Instruct)

- Chat-template tokenization consistent between `apply_chat_template(tokenize=True)`
  and tokenize=False + manual tokenization (42 tokens).
- K-sample generation, confidence elicitation, and hooked feature extraction all
  work: 831 feature columns, raw tensors `h_last/h_ans` (L+1, B, d),
  `gate_pre/sel/down` (B, L, 2, d_ff). `scripts/smoke_test.py` passes.

## Model/knowledge calibration on PopQA person sample

| model | prompt | K | frac correct | greedy correct | mean verbalized conf |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | zero-shot | 5 | 0.09 | 0.10 | 90.6 |
| Qwen2.5-1.5B-Instruct | few-shot | 5 | 0.08 | 0.12 | 92.1 |
| Qwen2.5-7B-Instruct | few-shot | 5 | 0.175 | 0.225 | 88.5 |

Takeaways: small models are confidently wrong on PopQA long tail; few-shot does not
change accuracy (but formats better); 7B roughly doubles accuracy. Hence the pilot
uses 7B and a head-heavy popularity mix to get label variance.

## Dataset (pilot)

- PopQA person relations: father, mother, place of birth, occupation
  (PopQA has no date of birth; substitution documented in PLAN.md).
- Head-heavy quintile mix (10/15/20/25/30% least→most popular): 1087 questions,
  1046 unique subjects: father 300, mother 187, place of birth 300, occupation 300.

## Full pilot run (completed 2026-09-29)

- `scripts/collect.py --model Qwen/Qwen2.5-7B-Instruct --k 30 --batch-questions 8
  --tag pilot7b --fewshot`
- Generation finished in ~70 min: 1087 questions, mean fraction-correct 0.170,
  greedy accuracy 0.193, mean verbalized confidence 90.9.
- The first feature-extraction pass was killed at 888/1087 questions by the
  session boundary; incremental saving + `--resume-feat` was added, and the rerun
  completed (~25 min): 1087 x 967 feature matrix, 5.5 GB raw tensors.
- Probe evaluation: `probe_eval.py --tag pilot7b --raw-probes` (frac label),
  `--label greedy`, plus `scripts/extra_probes.py` for combined-feature (H3) and
  targeted difficulty checks.

### Results (frac >= 0.5 label, n=1087, 16.5% positive)

A plain-English report with figures is in `report/REPORT.md` (`REPORT.html` for a
single shareable file, `REPORT.pdf` for printing).

| family | AUROC | best variant |
| --- | --- | --- |
| surface (difficulty) | 0.786 | s_pop etc. |
| verbalized confidence | 0.532 | 1-100 integer |
| hidden states | 0.871 | layer 18, answer pos |
| MLP summaries (17 features) | 0.863 | layer 2, last prompt |
| raw MLP `sel` (PCA-256) | 0.853 | layer 16 |
| **hidden + all MLP summaries** | **0.907** | 4536 features |

Historical bootstrap-mean deltas: MLP summaries - hidden = -0.008
[-0.041, +0.024]; hidden+MLPsums - hidden = +0.036 [+0.023, +0.051].
MLP summaries - confidence = +0.331. These are exploratory comparisons, not
equivalence or complete difficulty-control tests. Within-bin AUROC stays
0.79-0.90; verbalized confidence stays ~0.5 in every bin. Full tables and
interpretation in `results/FINDINGS.md`; per-run reports in
`results/pilot7b_report.md` and `results/pilot7b_greedy_report.md`.

## Quick pipeline checks (already done)

- `fs15` (1.5B, 60 q, K=5): probe_eval end-to-end works; numbers not interpretable
  (4 positives).
- Evaluation implemented: StratifiedGroupKFold(5) by subject, inner C selection,
  preprocessing fitted once per outer training fold, pooled OOF AUROC +
  bootstrap CI, layer curves, per-relation breakdown, popularity-quintile Stage 2,
  residual diagnostic, permutation check.
