# Confidence reporting: summary of everything done so far

**Keshav Krishna · New York University · 16 October 2026**  
Two-page summary. Fuller version: [`CAUSAL_CONFIDENCE_REPORT.md`](CAUSAL_CONFIDENCE_REPORT.md).

**The short version.** We asked whether a model can be trained to report how likely it is to answer correctly, and whether that report beats a strong predictor built from the model's own output probabilities.

1. **Yes on the readout test.** After fixing two evaluation bugs (below), a trained readout adds information beyond the probability baseline in **5 of 5 runs** — three seeds at 0.5B and two independent seeds at 7B, every interval above zero. The gain is modest (+0.012 to +0.045 Brier) and the report's own absolute score is still worse than the baseline's; it is complementary information, not better calibration.
2. **Not yet on transfer.** Reports fitted on the main knowledge variants do not transfer positively to an independently trained knowledge adapter: all five measurements are negative. A diagnosis and a balanced retest are in progress.
3. **Not yet on native reports.** When the model emits the number itself through small in-MLP adapters, the emitted number is currently constant even though the underlying distribution weakly tracks competence (correlation 0.13–0.26). Two implementation flaws (wrong KL suffix; decoding mismatch) were found and fixed; a clean fixed configuration is running.
4. **No for the earlier manipulation.** The MLP-attenuation branch failed its locked validation and is closed.

![Status dashboard](figures/knowledge_confidence/fig0_summary_dashboard.png)

## Scorecard

| Question | Status | Key evidence |
| --- | --- | --- |
| Do knowledge variants change competence cleanly? | **Supported** | 5/5 verification passes; spread 0.70–0.94; unexposed-fact success 0.08–0.14 |
| Does a trained readout add value beyond output probabilities (held-out subjects)? | **Supported** | added value +0.012 … +0.045, all 5 CIs above zero |
| Does the within-question pair objective help? | **Not supported** | T2 worse than T1 in all 5 studies |
| Does it transfer to a new knowledge variant? | **Negative so far** | transfer added value −0.085, −0.073, −0.062 (0.5B), −0.025, −0.009 (7B) |
| Can the model emit a trustworthy native number? | **Unresolved** | emitted constant; distribution tracks weakly; fixed config running |
| Can MLP attenuation test confidence training? | **No** | all three locked gates fail; branch closed |

## 1. The question

Given the *same question* posed to the *same base model* with different learned knowledge, can a small trained component predict this model's own success probability — and does that prediction beat what the model's output probabilities already tell us?

Three test requirements recur everywhere: identical question wording across states, labels from independently sampled and graded answers, and held-out-subject evaluation against the strongest probability baseline.

## 2. The foundation: controlled knowledge variants

Four counterbalanced adapters (rank-8, gate/up projections of a frozen Qwen2.5 model) each learned half of 240 invented facts, arranged so every fact is known by exactly two of four variants; a fifth adapter learned a disjoint fact pool. Training used five question phrasings and verification used a sixth, never seen. Sample results:

| Study | Cross-variant spread | Success on unexposed facts | Paraphrase survival |
| --- | --- | --- | --- |
| 0.5B seeds 0–2 | 0.70, 0.76, 0.70 | 0.13, 0.14, 0.12 | 76–82% |
| 7B seeds 0–1 | 0.94, 0.92 | 0.08, 0.08 | 88% |

This substrate is the project's most reusable asset: it changes what the model knows while keeping the question fixed, and it is verified by grading actual answers, not exposure counts.

## 3. Positive result: the trained readout adds information

A small MLP reads features at the answer fork (probability summaries and the model's own hidden state) and is trained against the model's measured success. Under the corrected evaluation (draw-level Brier, out-of-fold report predictions), adding it to the strongest probability baseline improves held-out Brier in every run:

| Study | Added value | 95% interval | Baseline Brier |
| --- | --- | --- | --- |
| 0.5B seed 0 | +0.0208 | [0.0143, 0.0270] | 0.184 |
| 0.5B seed 1 | +0.0120 | [0.0031, 0.0205] | 0.198 |
| 0.5B seed 2 | +0.0177 | [0.0095, 0.0265] | 0.177 |
| 7B seed 0 | +0.0452 | [0.0280, 0.0608] | 0.202 |
| 7B seed 1 | +0.0243 | [0.0150, 0.0327] | 0.198 |

A readout audit explains why earlier attempts failed: the answer-fork logits are exactly recoverable from the saved hidden state, and a cross-fitted probe on the **raw** hidden reproduces the probability baseline (reproduction MSE 0.00117 against a 0.001 gate; Brier 0.186 vs 0.184). The earlier failure was a lossy random projection plus in-sample predictions, not missing information.

**Precise claim:** *a trained readout improves a specified probability-based predictor on held-out subjects within the trained knowledge variants.* It does not yet show the model uses this information, nor that it generalizes to new variants.

## 4. Negative and unresolved results

![Toy summary](figures/knowledge_confidence/fig1_toy_rl_arms.png)

**Toy world (E1–E3, the first attempt).** Reward-trained arms improved retrieval (+2.6 points greedy over the reward-only arm) but a matched-supervision arm matched them, the report was worse than always saying 50%, and added value was ≤ +0.0005. Nothing here survived the stronger tests later.

**Paired objective fails everywhere.** The within-question contrast term is worse than plain pointwise training in all five corrected studies (−0.008 to −0.021 Brier). Whatever it is meant to buy, it does not show up on held-out subjects.

**Transfer is negative — with a caveat.** All five transfer measurements are negative, but the diagnosis matters: the transfer variant was trained on *all* its facts, so the transfer set had ~78% success against ~27% in training; the baseline-plus-report combination then extrapolated badly. The corrected seed-0 numbers (baseline 0.307, raw report 0.281, combination 0.392 Brier) show a *combination* failure, not proof that the report lacks transferable signal. A balanced retest (half the transfer facts, prevalence ~50%) is running.

**Native reports are unresolved.** Training adapters to emit the number themselves has so far produced constant emissions. The diagnostics show the report *distribution* carries a weak competence signal (expected-value correlation 0.18–0.26 across seeds) while the greedy emitted number stays at the floor — calibration, decoding, and optimization are unresolved, and capacity was explicitly not blamed. Two flaws were found and fixed: the answer-preservation KL was evaluated on the report suffix (nonzero-β runs were contaminated), and training scores whole strings while evaluation decoded token-by-token (now both readouts are reported, plus the third "argmax complete string"). A fixed configuration (corrected KL, single layer, rank 8) is running before any capacity sweep.

**Attenuation branch closed.** The MLP channel-damage manipulation failed all three locked gates (damage +0.0025 vs required 0.05; target-minus-control +0.0024 vs 0.03; restoration −0.08 vs 0.5) on 963 questions and 15,408 sampled answers.

## 5. Corrections to earlier claims

| Earlier claim | Correction |
| --- | --- |
| "The report adds no value beyond the baseline" | Artifact of in-sample report predictions (the augmented predictor overtrusted an overfit reporter) plus the lossy hidden projection. With out-of-fold predictions and raw hidden states, added value is positive in 5/5. |
| "The hidden state is uninformative" | Wrong under the tested readouts: the raw hidden reproduces the baseline exactly (logits are a frozen-head function of it). The 256-d projection was the problem. |
| "Draw-level Brier fixed the sign" | No: the variance term is common to both predictors. The reversal came from cross-fitting and the representation change. |
| Transfer "failed" | It combines a report problem with a combination/extrapolation problem; the balanced retest will separate them. |

## 6. Running now and next

- **Local:** readout-gain comparison (T1 vs T2 × probability-only vs hidden-inclusive) and the fixed native configuration with corrected KL and three decoding readouts; tracker armed.
- **Cluster:** balanced-transfer 7B study on the frozen main variants (job 20077531).
- **Next:** transfer readout (discrimination / calibration / combination separately), then a small **answer-versus-lookup** experiment with the external monitor — labeled externally controlled; a native metacognitive control version requires the model itself to make the decision.

## Where the evidence lives

- `runs/causal_confidence/cc_v1/knowledge/seed{0,1,2}/` — local knowledge studies (corrected evaluation).
- `runs/section33_v1/causal_confidence/cc_v1/know7b_seed{0,1}/` — 7B studies; `p2c/` — locked attenuation validation.
- `runs/causal_confidence/cc_v1/knowledge/diagnostics/` — readout audit, feature ablation, native sweeps.
- Figures: `report/figures/knowledge_confidence/` (dashboard `fig0`, toy `fig1`, P2 `fig2`, substrate `fig3`, reporter `fig4`, probes `fig5`, ablation `fig6`).
