# Pilot Plan — Hidden-State vs. MLP-Interaction Probes for Correctness on PopQA

Status: **historical design; pilot completed**. Evidence review: 2026-10-03.
The implemented procedure and limits are in `report/REPORT.md` and
`report/AUDIT.md`. Early tables below record proposed choices. The main run
uses 7B, all-layer raw storage, two token positions, and StratifiedGroupKFold.
The auxiliary attention model is not implemented.

This is the "preliminary experiment"
from the meeting notes, scoped to run end-to-end on the Strix Halo machine.

---

## 1. Question and hypotheses

**Pilot question.** Do features of the MLP *selection geometry* predict whether the
model will answer a factual question correctly, comparably to a standard
hidden-state probe, and does the signal survive popularity/difficulty control?

| # | Hypothesis | Operational test |
| --- | --- | --- |
| H1 | MLP-interaction features carry correctness signal | AUROC(MLP summary probe) > AUROC(surface-feature probe) and > 0.5 |
| H2 | MLP-interaction probe is comparable to hidden-state probe | \|AUROC_MLP - AUROC_hidden\| <= 0.03 at matched layers; no significant paired difference |
| H3 | MLP features add signal beyond hidden states | AUROC(hidden + MLP) - AUROC(hidden) > 0 |
| H4 | Signal is not purely popularity/difficulty | within-s_pop-bin AUROC of MLP probe remains > 0.5; probe correlates with the residual of fraction-correct after regressing out surface features |

**Decision rule (from meeting notes).** If H2 holds, the direction is worth
pursuing; if the MLP probe drops substantially (> 0.05 AUROC vs. hidden-state
baseline), treat the direction as weak for now. H3/H4 decide whether a full study
is justified.

**Theoretical caveat.** `gate_pre = W_gate x` is linear in normalized MLP
input `x`. If `W_gate` has full column rank, unrestricted linear classifiers
on either vector have the same function class. Rank is not measured here.
Finite samples and regularization can change their test scores. The pilot's
post-block hidden state is not `x`, and PCA changes the raw probe. This
observation does not prove equal AUROC or establish that nonlinear summaries
are necessary. The proposed design explores summaries and cross-layer features.

---

## 2. Scope decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Model (primary) | `Qwen/Qwen2.5-1.5B-Instruct` | Small enough to iterate, strong factual recall for popular entities, 28 layers, d_model 1536, d_ff 8960; used in the geometric-recall paper's sweep (0.5B variant). |
| Model (pipeline test) | `Qwen/Qwen2.5-0.5B-Instruct` | Fast smoke tests; validates code paths before the real run. |
| Dataset | PopQA (`akariasai/PopQA`), person relations | Support of the main experiment. |
| Relations (pilot) | `father`, `mother`, `place of birth`, `occupation` | Person subjects; `occupation` is the popularity-linked relation. **PopQA has no `date of birth` relation** — substitute `place of birth` for the pilot; add Wikidata `P569` later if the main study needs it. |
| Sample size | ~1000 questions, stratified over `s_pop` quintiles within relation | Enough for 5-fold grouped CV with ~200 test items per fold; keeps generation cost manageable. |
| Correctness labels | K=30 samples, T=1.0, top_p=1.0, `max_new_tokens=16`; fraction correct | Mirrors the main experiment; K configurable (`--k 10` for fast pass). |
| Primary binary label | `frac_correct >= 0.5` | Also report `frac_correct` regression and greedy-correct. |
| Confidence report | separate generation, 1-100 integer | Mirrors the main experiment; comparator, not a probe input. |
| Evaluation split | `GroupKFold(5)` grouped by `subj_id` | Same subject appears under multiple relations; avoid leakage. Same established protocol as the geometric-recall paper (subject-disjoint splits). |
| Probes | standardized L2 logistic regression with nested C selection; MLP probe as nonlinear robustness check | Fair comparison across feature sets with different dimensions. |

---

## 3. Feature families

All features collected in a **single forward pass** on `prompt + gold answer`
(causal model: positions up to the last prompt token are unaffected by the answer).
Two capture positions: `last_prompt` (the token before the answer starts — the
"can I answer this?" state) and `answer` (first gold-answer token — the "writing the
answer" state).

### S — surface / difficulty (baseline predictor)
`log10(s_pop)`, `log10(o_pop+1)`, relation one-hot (4), number of gold aliases,
prompt token length, subject token length, name-collision proxy
(count of PopQA subjects sharing the subject's last name), answer-set size.

### P — model probability baselines
`p_gold` at last prompt, next-token entropy, top-1 probability, top-1/top-2 margin,
gold rank (capped at 100). These are strong baselines: a probe must beat `p_gold`.

### H — hidden-state baseline (per layer, both positions)
- `h_last[l]`: residual stream after layer l at the last prompt token (1536-d).
- `h_subj[l]`: residual stream after layer l at the last subject token (1536-d).

### M — MLP interaction features (per layer, both positions)
Captured tensors: `gate_pre` (`gate_proj` output), `sel` (`down_proj` input),
`down` (`down_proj` output).

- **M-summary (nonlinear distribution stats)**, per layer:
  entropy of `softmax(gate_pre)`; entropy of `softmax(gate_pre^2)`; alignment
  entropy (normalized); active fraction (`gate_pre > 0`); top-5 mass; Gini; max;
  kurtosis; `||sel||`, `||down||`, `||down|| / ||h||`, `cos(down, h)`.
- **M-answer alignment**, per layer:
  `cos(down, D_gold)`; `||proj_{D_gold}(down)|| / ||down||`; MLP margin
  contribution `<down, D_gold> - max_{v in C} <down, D_v>` where `C` = top-50
  competitor tokens of the model's next-token distribution (plus gold token);
  entropy of `softmax(D_C down)`.
- **M-raw** (for the direct comparison): `gate_pre`, `sel`, `down` vectors at a
  subset of layers (every 4th layer plus best hidden-probe layer), probed with
  ridge/PCA-256 logistic + MLP probe.

### C — prompted confidence (comparator)
Integer 1-100 confidence from a separate prompt; used in Stage 2 (difficulty
control) as the "standard prompted confidence score".

---

## 4. Pipeline

```
prepare_data.py    PopQA -> person-relation subset -> s_pop-stratified sample
                   -> data/popqa_person.jsonl (prompt, aliases, features S)

collect.py    (GPU) per question:
                 1) sample K answers (T=1.0)     -> fraction_correct, greedy_correct
                 2) confidence prompt (greedy)   -> confidence_1_100
                 3) feature pass on prompt+gold   -> H, M summaries, M raw (subset),
                                                    P baselines
               -> results/labels.csv, results/features_summary.parquet,
                  results/raw/M_*.npz (subset of layers)

probe_eval.py      GroupKFold CV probes -> AUROC tables, layer curves,
                   paired comparisons, figures
                   Stage 2: popularity bins, residual diagnostics, confidence
                   comparator

report.md          auto-generated summary answering H1-H4 with the decision rule
```

## 5. Metrics and reporting

- Mean±std AUROC across the 5 grouped folds; pooled out-of-fold AUROC with
  bootstrap 95% CI; paired DeLong test between the best hidden-state and best MLP
  probes.
- Spearman correlation between probe scores and `frac_correct`.
- Layer-by-layer AUROC curves for H and M (two panels), plus best-layer markers.
- Per-relation breakdown table.
- Stage 2: within-bin AUROC for (surface predictor, MLP probe, hidden probe,
  prompted confidence); residual diagnostic (regress `frac_correct` on S, correlate
  residuals with probe scores and confidence).
- Sanity checks: label permutation test (AUROC ~0.5 when labels shuffled), check
  that the surface predictor is strong for `s_pop` (as expected), and that `p_gold`
  is a strong correctness predictor.

## 6. Acceptance criteria and interpretation

| Outcome | Interpretation | Action |
| --- | --- | --- |
| MLP probe ~ hidden probe (H2), and H3/H4 hold | Signal present, direction viable | proceed to full architecture (cross-layer attention pool over gate geometry) with difficulty-controlled training |
| MLP probe well below hidden probe (Δ > 0.05) | Gate-geometry summaries lose the signal | try raw `sel` probes / nonlinear pooling before abandoning; if still weak, abandon |
| Raw `sel` probe >> summary probe | signal is in the write geometry, not scalar concentration | redesign auxiliary model to operate on vectors |
| MLP features ≈ surface features only | probe just relearned popularity | abandon or redesign with difficulty-balanced training |

## 7. Risks / open issues

1. **Sample size vs. probe dimension.** Raw 8960-d probes overfit; handled with
   ridge/PCA/nested C and MLP probes, and by treating raw probes as exploratory.
2. **Label noise.** K=30 binomial labels at the 0.5 threshold; report regression
   targets and greedy labels as robustness.
3. **GPU compatibility.** ROCm on gfx1151 (Strix Halo) can need
   `HSA_OVERRIDE_GFX_VERSION=11.0.0` or `PYTORCH_ALLOC_CONF=expandable_segments:True`.
   Smoke test covers this.
4. **PopQA date-of-birth gap.** Documented; `place of birth` substituted for the
   pilot. Adding `P569` from Wikidata is a small follow-up if needed.
5. **"Interaction" definition drift.** The meeting notes mix ROME's W2-column story
   with the geometric paper's gate story. We report both feature families
   separately so the result is interpretable under either view.

## 8. Compute estimate

- Generation: 1087 questions x 30 samples x ~12 new tokens at T=1.0, batched
  (8 questions = 240 sequences per batch). On the Strix Halo iGPU with
  Qwen2.5-7B this is projected at ~60-90 min (measured 1.25 q/s at K=5, batch 4,
  Qwen2.5-7B).
- Feature pass: 1087 forward passes over ~70-token sequences, plus the batched
  `W2`-alignment matmuls — tens of minutes.
- Probe CV: ~150 logistic probes, ~20s each worst case, plus a few MLP probes —
  under an hour on CPU.

## 9. Pilot run decisions (frozen 2026-09-28)

These were set after the environment and pipeline smoke tests; deviations from the
original sketch are marked.

- **Hardware/software.** AMD Strix Halo (Radeon 8060S, gfx1151), PyTorch
  `2.10.0+rocm7.13.0a20260513` from the gfx1151 wheel index
  (`https://rocm.nightlies.amd.com/v2/gfx1151/`). The standard ROCm 6.4 PyTorch
  wheel has **no gfx1151 kernels**; `HSA_OVERRIDE_GFX_VERSION=11.0.0` does not
  work. See `scripts/setup_env.sh`.
- **Model.** `Qwen/Qwen2.5-7B-Instruct` for the pilot, `Qwen2.5-0.5B/1.5B` for
  pipeline tests. The 1.5B was unusable as a knowledge source here: 9% fraction
  correct with 90+ verbalized confidence on a first-pass sample.
- **Prompt.** Few-shot chat prompt with 4 short-answer exemplars (one per relation).
  Zero-shot and few-shot gave the same 1.5B accuracy, but few-shot improves output
  format compliance.
- **Sampling (deviation).** Head-heavy popularity mix instead of uniform per-bin:
  within each relation, questions are drawn from popularity quintiles with weights
  10/15/20/25/30% (least -> most popular). The uniform-tail mix yielded ~9%
  fraction-correct, too imbalanced for stable probe CV; the head-heavy mix targets
  label variance while preserving a popularity gradient for Stage 2.
- **Sample.** 1087 questions (father 300, mother 187, place of birth 300,
  occupation 300; 1046 unique subjects).
- **Labels.** Primary binary target `frac_correct >= 0.5` over K=30 samples at
  T=1.0, plus deterministic `greedy_correct` as a secondary target (more positives:
  ~22% vs ~17% on the 80-question calibration subset).
- **Feature positions.** Last prompt token ("can I answer this?") and first gold
  answer token (teacher-forced) in a single pass. Subject-token features from the
  geometric paper are a follow-up, not in this pilot.
- **Answer-alignment features** use the gold token and the top-50 tokens of the
  model's own next-token distribution as competitors; MLP write alignment is
  computed via `W2`-column projections onto unembedding rows.
- **Raw MLP probes** (PCA-256) are run for 5 layers and 3 tensors as the
  exploratory "does the geometry itself carry signal" check.

## 10. Follow-up design (only if the pilot supports the direction)

The pilot tests *signal existence* with per-layer probes. If H2/H3 hold, the
auxiliary model from the meeting notes can be implemented as:

- **Input.** Per layer `l`, a feature token built from the MLP selection geometry:
  a learned linear projection of `sel[l]` (the gated value write; PCA-initialized or
  trained end-to-end) concatenated with the 17 scalar summary stats. This yields a
  sequence of 28 tokens (one per layer) of dimension ~64-128.
- **Pooling.** 1-2 transformer encoder layers with a learnable CLS token, or
  cross-layer attention (CLAP-style) over the layer tokens; CLS -> linear scalar.
- **Targets.** Correctness (BCE) as the primary target; optionally the K-sample
  fraction-correct as a regression target; verbalized confidence as an auxiliary
  distillation target.
- **Difficulty control during training.** Stratify batches by `s_pop` quintile,
  or add a difficulty-adversarial loss (probe must be uninformative about the
  difficulty bin). This mirrors the difficulty-binning logic of the main
  experiment and prevents the auxiliary model from returning to a popularity
  shortcut.
- **Key evaluation (the actual research claim).** Compare three confidence
  estimates on held-out data: (a) prompted verbalized confidence, (b) the CLS
  auxiliary model, (c) a scalar readout of the CLS trained with a
  difficulty-adversarial objective. Report within-bin AUROC (which does not
  advantage difficulty-tracking scores) and the Spearman correlation between
  each score and the surface-difficulty residual. The claim is supported if (c)
  keeps within-bin AUROC while losing residual correlation with popularity.

Implementation status: not started; depends on the pilot verdict.
