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

## 15 October 2026 — causal confidence training plan implementation

Added `causal_confidence/`, the implementation of
`report/CAUSAL_CONFIDENCE_TRAINING_PLAN.md`. This entry records the code, not
new results; no 7B stage has been run yet.

- `protocol.py` freezes the model (`Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a3545...`), audited relations, prompts, the 21-value report grammar,
  sampling policy, partition sizes, six condition slots, mixture weights,
  pair types, and the P0-P3 gates.
- `data.py` builds subject-disjoint partitions from the section 33 cohort and
  the PopQA source, with dedupe, reference audits, and a hashed manifest.
- `episodes.py` and `interventions.py` implement retained-state branching,
  middle-layer MLP channel attenuation, restoration, matched random and
  other-group controls, and the contract checks (cache/full-run equivalence,
  branch isolation, adapter gating, frozen parameters).
- `outcomes.py` writes immutable correctness banks with the frozen matcher;
  `reporters.py` provides the scalar diagnostic and the rank-8 native report
  adapter with constrained greedy decoding; `objectives.py` implements the
  point/pair losses, exact grid expectation, and the sampled REINFORCE
  estimator; `baselines.py`, `evaluate.py`, and `mechanisms.py` cover
  declared-information baselines, Brier/contrast/cluster bootstrap, and
  report-branch interventions.
- P0 regenerates the toy diagnostic rows (no toy checkpoints were retained)
  and runs the reproduction/correctness positive controls. P1 checks the
  episode contract and audits the discovery answer policy. P2 runs bounded
  discovery and one locked validation. P3-P6 are gated stubs.
- Tests: `tests/test_causal_confidence.py` implements the required
  implementation checks, including exhaustive-enumeration agreement for the
  grid expectation, policy-gradient agreement on a tiny categorical, cache
  equivalence with a last-layer diagnostic, adapter isolation, frozen
  parameters, Brier shapes, partition isolation, and the leakage scan.

Cluster sequence: `causal_confidence/cluster/01_p0.sbatch` through
`04_p2_validate.sbatch`; see `causal_confidence/cluster/README.md`.

## 16 October 2026 — controlled knowledge-variant study (revised protocol)

After the attenuation route showed no headroom, the revised experiment built
several versions of the same model with different factual knowledge and asked
whether a shared confidence reporter can track which version knows what.

**Substrate (verified).** A synthetic corpus of invented facts (hash-based
values, 12-value pools; four attributes) with one training question and six
paraphrases each. Four knowledge variants are rank-8 LoRA adapters on
gate/up projections, trained on a balanced incomplete block design: every
fact is learned by exactly two of four variants (each pair shares 1/6), so
each variant knows ~half the questions and question wording is identical
across versions. A fifth adapter on disjoint subjects/facts is the transfer
variant. Verification trains on five templates, holds out a sixth, and gates
on cross-variant spread, paraphrase survival, format rate, reproducibility,
and success on unexposed facts (the last added after an earlier generator
let variants infer unexposed facts). Three local seeds (Qwen2.5-0.5B) and one
7B pilot all pass: spread 0.70-0.94, unexposed-fact success 0.12/0.08,
format <= 0.008.

**Reporter (negative).** Grid-mean heads over retained-state features were
trained pointwise (T1) and with the within-question pair term (T2), inputs
standardized with train statistics. On held-out subjects the strongest
ordinary baseline is the prefix-probability logistic (test Brier 0.103-0.121
at 0.5B, 0.146 at 7B). The reporter matches it only when restricted to
question+prefix (0.106) and adds nothing (+0.0001); the paired objective
gives a marginal contrast improvement without incremental value, and is
worse than T1 on the main test in all three local seeds and at 7B.

**Readout probes (negative).** The seeded 256-d hidden projection is not the
bottleneck: the raw 896-d final suffix hidden state is uninformative under
ridge (test MSE 0.255) and a regularized MLP with dev-selected weight decay
(0.246), both at the constant-prediction level, versus 0.103 for the prefix
probability features. The rich top-k prefix block is worse than the summary
prefix block. No tested readout (linear or shallow nonlinear) beats the
model's own answer-fork probabilities.

**Interpretation.** On this substrate the usable competence signal is
already in the output distribution; trained confidence reports add no
incremental predictive value beyond the tested probability baselines (H3
fails), and the pair objective does not improve the head family. The
attenuation branch (P2) was still running its locked validation when this
was written; its verdict is reported separately.

Artifacts: `runs/causal_confidence/cc_v1/knowledge/seed{0,1,2}/`,
`runs/causal_confidence/cc_v1/know7b_seed0/`, and the ablation/probe outputs
in `scripts/knowledge_feature_ablation.py` and
`scripts/knowledge_readout_probe.py`.

## 16 October 2026 — P2 locked validation: attenuation branch closed

The p2c validation finished: 963 validation items, 15,408 draws, six slots
plus five matched-random replicates and norm/other-group controls, gates
unchanged from the frozen protocol. All three prespecified gates fail.

| Gate | Effect | Threshold | Result |
| --- | --- | --- | --- |
| Target damage (strong) | +0.0025 [-0.0005, +0.0057] | >= 0.05 | fail |
| Target minus matched random | +0.0024 [-0.0009, +0.0057] | >= 0.03 | fail |
| Restoration recovery | -0.077 | >= 0.5 | fail |

Clean success is 6.0% on this pool. Per-relation damage is within noise
(father +0.28 pts, mother -0.15 pts, place of birth +0.57 pts), the
unrelated-task check shows no damage (-0.15), and the matched-random and
other-group controls are indistinguishable from the target. The attenuation
manipulation has no usable effect, so the intervention-dependent training
branch (P3-P6) stops here, per the plan's decision table. With the
knowledge-variant study also negative on added value (previous entry), the
current evidence supports neither causal-supervision route for incremental
confidence value.

Artifacts: `runs/section33_v1/causal_confidence/cc_v1/p2c/` (validation JSON,
per-question outcomes, damage/restoration and per-question figures).
