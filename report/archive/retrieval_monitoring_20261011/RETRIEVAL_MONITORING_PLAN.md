# Retrieval-Monitoring Plan (revised)

Status: proposed. Date: 11 October 2026. Model: Qwen2.5-7B-Instruct.
Source proposal: `report/RETRIEVAL_MONITORING_EXPERIMENTS.md`.
Frozen evidence: `runs/section33_confirmatory/`.

This plan applies the review fixes and stages the work from small tests to
large tests. Each stage has a gate. Do not start a later stage until the
earlier gate passes.

---

## 1. Claims

Three claims are separate. Every table must say which claim it supports.

| Claim | Statement | Test |
| --- | --- | --- |
| **C1 native** | The model's own confidence responds to private retrieval damage. | Stage M1 |
| **C2 learned** | A small learned reporting module can respond to private retrieval damage. | Stage M2 |
| **C3 causal** | A specific pathway carries the reliability signal into the report. | Stage M3 |

The first paper claim is C1. C2 is a distillation result. C3 is a mechanism
result. Do not mix them.

## 2. Review fixes

| Fix | Implementation |
| --- | --- |
| P1 separate claims | Section 1. C1, C2, C3 are named. |
| P2 noisy reliability target | Stage M1 uses 8 draws and a hierarchical estimate. The primary endpoint is the paired slope of confidence change on reliability change. |
| P3 anomaly versus reliability | Section 5 lists the floor-effect control, the matched first-order control, the random-map control, and the cross-relation map control. |
| P4 selectivity gate | Stage S1.3 tests relation-map attenuation on development subjects. Stage M1 stops if the gate fails. |
| P5 power | Section 3.3 gives the power table. Stage M1 uses 320 to 500 subjects. |
| P6 two text regimes | Stage S0 reports question-only and question-plus-candidate baselines. |
| P7 paths and fork code | The run path is `runs/section33_confirmatory`. `section33/analysis/retained_state.py` implements the fork. `tests/test_retained_state.py` checks it. |
| P8 pathway power | Stage M3 pre-simulates the equivalence margin before any run. |
| P9 toy world scope | Stage S2 is a debug world only. It cannot support claims about pretrained models. |
| P10 related work | Section 8 adds the missing references. |

## 3. Small stages

### 3.1 Stage S0: input-only baselines (done)

Question: can a text-only observer explain the saved probe advantage?

Setup. The models use the frozen train split and score the frozen test split.
The prospective regime sees the question, relation, and popularity. The
after-answer regime also sees the candidate answer. Metrics are overall AUROC
and mean within-question AUROC on the 341 mixed-outcome test questions.

| Track | Predictor | Overall AUROC | Within-question AUROC |
| --- | --- | ---: | ---: |
| A | Question only | 0.796 | n/a |
| A | Saved H (hidden probe) | 0.888 | n/a |
| B | Question only | 0.788 | 0.500 |
| B | Question + candidate | 0.870 | 0.695 |
| B | Saved H (hidden probe) | 0.935 | 0.832 |
| B | Saved P (probability probe) | 0.838 | 0.738 |
| B | Saved V (stated confidence) | 0.571 | 0.572 |

The Track A delta is `text - H = -0.092 [-0.114, -0.071]`. The Track B delta
is `text - H = -0.065 [-0.082, -0.051]`. Both intervals exclude zero. The
question-only within-question value is 0.500 by construction: all samples of a
question receive the same question score.

Outputs: `runs/section33_confirmatory/evaluation/text_baselines.json` and
`text_baselines_scores.parquet`. Code: `section33/analysis/text_baselines.py`,
`scripts/run_text_baselines.py`.

**Gate S0: passed.** The strongest feasible input-only observer is 0.065 AUROC
below the hidden probe overall and 0.137 below it within questions. A private
state test is justified. This is not proof of privileged access. The hidden
state still contains the model's own answer computation.

### 3.2 Stage S1: protocol integration

This stage has no model science. It proves that the apparatus works.

**S1.1 Power simulation.** Use `section33/analysis/power.py` patterns. The
paired t-test sample size is:

| Paired SD of confidence change | 3 pts | 5 pts | 7.5 pts | 10 pts |
| --- | ---: | ---: | ---: | ---: |
| 10 | 88 | 32 | 14 | 8 |
| 15 | 197 | 71 | 32 | 18 |
| 20 | 349 | 126 | 56 | 32 |
| 25 | 546 | 197 | 88 | 50 |

The table uses two-sided alpha 0.05 and power 0.80. It counts subjects, not
answers. A subject contributes both relations, so pooling can reduce the
effective variance. Use the table for planning only. Estimate the paired SD
from the S1.3 dry run.

**S1.2 Retained-state integration test.** Run the fork on Qwen2.5-0.5B
Instruct and on Qwen2.5-7B-Instruct. Compare a prefix-fork run with an
independent full run. The independent run must match to float tolerance. The
unit test does this on a tiny random model. The integration test does it on the
real model with the real prefix and the real suffixes. The check passed on
Qwen2.5-0.5B in float32: max logit difference 8.6e-5, top-1 agreement 1.0,
and the intervention reached both branches. The 7B float16 check remains.

Implementation finding from the unit tests: a continuation reads cached keys
and values, not the fork-point hidden state. An intervention must run at a
decoder layer whose later layers recompute keys and values from the changed
state. A hook on the last decoder layer never reaches the branches. Use a
middle layer. The plan uses layers 8 to 20 in Qwen2.5-7B.

**S1.3 Relation-map selectivity dry run.** Use 40 development subjects and two
relations per subject. Fit relation readouts on training subjects. Apply four
conditions: identity, queried-relation attenuation, unqueried-relation
attenuation, and a norm-matched random direction. Measure the answer-branch
success of 8 draws per condition. Require the queried condition to reduce
success more than the unqueried and random conditions. Also fit a readout on
random labels and require no selective damage.

**Gate S1: the fork matches independent runs, and the intervention is
selective in the dry run.** If selectivity fails, return to Stage S2 and fix
the intervention. Do not run Stage M1.

### 3.3 Stage S2: toy world

Purpose: debug the full protocol in a controlled model.

Setup: 512 artificial entities, four random relation bijections over a shared
attribute vocabulary, dimension 64 or 128, one-layer transformer, three seeds.

Run the same four conditions, the floor-effect control, and the matched
first-order control. Fit an external reliability readout and a small reporting
module as separate methods. Train monitoring on one corruption family and test
on an unseen family and fresh bijections.

**Gate S2: the retained state contains a reliability signal beyond the
controls, and the signal transfers to new subjects and new corruptions.** A
probe-only result is acceptable here. It motivates Stage M2.

## 4. Major stage M1: native report (C1)

### 4.1 Design

- **Subjects.** 320 to 500 fresh subjects. Two relations per subject.
- **Splits.** Fit relation maps on training subjects. Select layers, ranks, and
  magnitudes on development subjects. Evaluate on test subjects. The old
  Section 3.3 test set stays exploratory and is not the new holdout.
- **Conditions.** Four per relation query: identity, queried-relation
  attenuation, unqueried-relation attenuation, and norm-matched random. Each
  condition gets the same visible tokens.
- **Retention.** Run the prefix once per condition. Fork deep copies into the
  answer branch and the confidence branch. Use fixed suffixes. Neither branch
  sees the intervention label.
- **Draws.** 8 answer draws per condition for the initial screen. Use 16 draws
  for a smaller confirmatory subset if the effect is close to the gate.
- **Target.** `p(q, z)` is the success probability of the answer branch under
  the fixed decoding policy. Estimate it with 8 draws and a beta-binomial
  shrinkage model. Treat the answer branch as the policy of record.

### 4.2 Endpoints

- **Primary.** Paired sensitivity. Regress the change in confidence on the
  change in `p(q, z)` across conditions, within subject and relation. The
  queried-relation slope must be larger than the unqueried-relation slope.
- **Secondary.** Overall AUROC and ECE of native confidence across conditions.
  Within-popularity-stratum AUROC.
- **Manipulation check.** The queried attenuation must reduce `p(q, z)` more
  than the unqueried and random conditions. Report the difference and its
  interval.

### 4.3 Controls

| Control | Purpose |
| --- | --- |
| Floor effect | Apply the same attenuation to questions the model already answers incorrectly. A confidence drop here indicates generic anomaly. |
| Matched first-order | Match conditions on the change in answer entropy or margin. Compare confidence at matched first-order uncertainty. |
| Random map | Fit the readout on random labels. Attenuation must not damage recall or move confidence. |
| Cross-relation map | Use the map of a different relation. Tests the map, not the query. |
| Clean replay | Report the same confidence task in a fresh pass. The retained-state branch and the replay must differ when the intervention is active. |
| Prompt anomaly | Add odd but harmless wording. Analyze separately because the visible text changes. |
| Report format | Ask confidence in a second format. The effect must not depend on one phrasing. |

### 4.4 Gate M1

Run Stage M2 if the native report tracks `p(q, z)` beyond the controls, or if
the external probe does. A probe-only positive result supports C2 with a
learned module. A native-only positive result supports C1. Report both.

## 5. Major stage M2: learned report (C2)

Only after Gate M1.

- **Data.** 500 to 1,000 training questions, 4 or more private states per
  question, 8 to 16 draws per state.
- **Arms.** Pointwise correctness supervision; hidden-probe distillation;
  pointwise plus contrastive ranking across random questions; pointwise plus
  ranking across private states of the same question. Add a no-state arm and a
  noise-ceiling arm.
- **Adapter.** Freeze the answer model. Activate the reporting adapter only in
  the confidence branch. This is a learned reporting module, not a changed
  answer policy.
- **Holdouts.** One intervention family is unseen. Untouched natural questions
  form a separate test. Use three seeds in the toy world. One real-model seed
  is a feasibility result.

**Gate M2: same-question state pairing beats ordinary supervision and random
pairing on held-out subjects, an unseen corruption family, and untouched
questions.** Gains on only the trained perturbation pattern are not enough.

## 6. Major stage M3: pathway (C3)

Only after Gate M2 or a strong native result.

- **Pairs.** 50 to 100 development pairs and 50 to 100 fresh evaluation pairs.
- **Method.** Identify candidate intermediate states between the perturbed
  subject state and the report. Patch a small candidate subspace. Compare with
  random subspaces of matched rank and norm. Test ablation, donor transfer, and
  restoration. Measure the answer distribution in a separate branch.
- **Power.** Pre-simulate the equivalence margin. If a tight margin is not
  identifiable at the planned size, report inconclusive and plan a powered run.
- **Control.** A first-order readout circuit can mimic parts of the pattern.
  Report answer entropy and margin changes with every result.

## 7. Analysis rules

- Predeclare splits, strengths, primary contrasts, and a minimum useful effect
  on development data.
- Cluster uncertainty by subject. Keep all relations, conditions, and answers
  of a subject together.
- Use paired contrasts and report multiplicity.
- Report eligibility and selection rates for every subset.
- Do not report calibration on a balanced matched pair set as population
  calibration.
- Keep layer, rank, and magnitude selection inside development.

## 8. Related work additions

- Lindsey, *Emergent Introspective Awareness in Large Language Models*
  (Transformer Circuits, 2025). The intervention-detection baseline.
- Lederman and Mahowald, *Dissociating Direct Access from Inference in AI
  Introspection* (arXiv:2603.05414). Detection and inference are separate.
- *Latent Introspection: Models Can Detect Prior Concept Injections*
  (arXiv:2602.20031). Underelicited detection.
- Macar et al., *Mechanisms of Introspective Awareness* (arXiv:2603.21396).
  The evidence-carrier-to-gate mechanism hypothesis.
- Singh, Linzen, Ravfogel, *Can LLMs Introspect? A Reality Check*
  (arXiv:2605.26242). The privileged-access and second-order conditions.
- Xi et al., *Calibration, Not Answer Selection: Distilling Internal
  Confidence* (arXiv:2609.33290). The nearest distillation work.

## 9. Implementation map

| Item | Path | Status |
| --- | --- | --- |
| Input-only baselines | `section33/analysis/text_baselines.py` | done |
| Baseline CLI | `scripts/run_text_baselines.py` | done |
| Retained-state fork | `section33/analysis/retained_state.py` | done |
| Fork unit tests | `tests/test_retained_state.py` | done (7 tests) |
| Fork integration check | `scripts/check_retained_state.py` | done (0.5B, float32) |
| Baseline results | `runs/section33_confirmatory/evaluation/text_baselines.json` | done |
| Baseline scores | `runs/section33_confirmatory/evaluation/text_baselines_scores.parquet` | done |
| Relation-map attenuation | `section33/analysis/relation_interventions.py` | next |
| Retained-state collector | `section33/models/collect_retained.py` | next |
| Pilot runner | `scripts/run_retrieval_monitoring.py` | next |

Commands:

```bash
.venv/bin/python -B scripts/run_text_baselines.py \
    --run runs/section33_confirmatory --track B --track A
PYTHONPATH=. .venv/bin/python -B -m unittest tests.test_retained_state -v
```

## 10. Stage summary

| Stage | Size | Compute | Gate |
| --- | --- | --- | --- |
| S0 baselines | saved data | done | passed |
| S1.1 power | CPU | minutes | plan sized |
| S1.2 fork integration | 0.5B and 7B | small | exact match (0.5B passed) |
| S1.3 selectivity dry run | 40 subjects | small | selective damage |
| S2 toy world | synthetic | CPU or one GPU | transfer to new corruptions |
| M1 native pilot | 320-500 subjects | medium | report tracks reliability |
| M2 learned report | 500-1,000 questions | large | beats matched arms |
| M3 pathway | 50-100 pairs | medium | selective loss of monitoring |

This plan follows ASD-STE100 principles: short sentences, active voice, and
one idea per sentence. It is not a certified Issue 9 text.
