# Can a model tell when its factual recall is unreliable?

Consolidated research report and experiment plan • 11 October 2026

Status: proposed research, informed by repository results and an implementation review. This is the authoritative report. Earlier versions are preserved in `report/archive/retrieval_monitoring_20261011/`. Updating this report does not repair the code or complete the pending experiments.

## 1. The direction in simple words

Quietly weaken a model's access to a particular fact, ask exactly the same question, and check whether the model knows that its answer has become less reliable. If it cannot report this naturally, test whether a small reporting adapter can learn to do so.

For example, weaken information used to recall a person's birthplace. Confidence about birthplace should fall when birthplace recall worsens. Confidence about occupation should stay stable when occupation recall stays stable. Equally large perturbations should produce different confidence changes when they have different effects on recall.

The research question is whether confidence tracks the consequences of a private change in retrieval, beyond the tested question-based, output-uncertainty, and generic perturbation baselines. The distinctive training idea is to pair the same question under different internal states, so visible question difficulty cannot explain differences between the paired targets.

The recommended first investment is an apparatus check and a 40-subject selectivity pilot. Large collection and adapter training should wait until the intervention demonstrably changes the intended retrieval process.

## 2. Inspiration, novelty, and claim boundaries

Geometric Factual Recall in Transformers [1] separates relational storage and selection in a controlled transformer setting. Its real-model experiments find linearly decodable answer information in subject-token states. This motivates interventions on relation-specific representations. It does not establish that Qwen implements the toy mechanism, or that a learned linear readout identifies a causal storage location.

Can LLMs Introspect? A Reality Check [2] distinguishes access to private information from stronger evidence of second-order monitoring. A model can exploit input cues or detect an unusual activation without estimating whether it can answer correctly. This motivates identical-text trials, external observers, perturbation controls, and a later pathway experiment.

Probe distillation [3], within-question confidence evaluation [4], and intervention-detection mechanisms [5] have close precedents. Novelty remains provisional. The candidate contribution is relation-specific monitoring that transfers to unseen internal interventions, with matched controls and a clearly defined answer policy.

Keep three claims separate:

- C1, native reporting: the untrained confidence channel carries useful information about changes in its own retrieval reliability.
- C2, learned reporting: a reporting adapter learns this behavior and transfers beyond its training interventions.
- C3, causal pathway: an upstream pathway contributes to reliability reporting while factual answering and basic reporting ability are preserved within declared tolerances.

External probe success establishes readable information, not C1 or C2. A learned adapter establishes C2, not native introspection. Even C3 supports a bounded functional account; it does not settle strong introspection or consciousness.

## 3. What the repository currently supports

The October 9 Section 3.3 confirmatory report records hidden-probe AUROC of approximately 0.888 before answering and 0.936 after answering. The stated-confidence probe reaches approximately 0.592 and 0.571. The tested MLP additions improve the baseline by approximately 0.0038 before answering and 0.0000 afterward; their intervals exclude the registered useful gain of 0.020. This limits the tested feature families, rather than proving that all MLP information is useless.

An exploratory analysis of saved after-answer predictions found mean within-question AUROC of 0.832 for the hidden probe, 0.738 for probability features, and 0.572 for the stated-confidence probe. It uses the 341 of 4,000 test questions with both correct and incorrect samples. These selected, after-answer results do not establish prospective monitoring or exclude answer-content shortcuts. The old test set is exploratory for this project; use a fresh final holdout.

Two newer pieces of work require correction before they support a gate:

- Text baselines: the loader used 2,300 training subjects despite a 2,000-subject manifest, mixed collections, and produced inconsistent draw counts. No train–test subject overlap was found, but the population and weighting mismatch invalidates the claimed frozen-cohort comparison. The current JSON preserves only Track B. Its question-plus-candidate AUROC of about 0.870 is provisional; the quoted Track A result is not preserved there. S0 has not passed.
- Retained-state code: seven existing tests pass, but continuation replaces retained attention masks with all ones. A padded-prefix diagnostic differed from the correctly masked full run by about 0.0351 in maximum logit value. The smoke script uses a complete QA prompt and supplied answer/confidence strings, rather than the proposed prospective interface. Cache primitives are tested; experimental integration remains pending.

The practical implication is to study how available reliability information reaches a report, while first repairing the evidence and apparatus checks below.

## 4. One protocol for the real-model experiments

Use Qwen2.5-7B-Instruct as the initial real model. Define p(q,z) as the probability that its answer branch answers question q correctly under concealed intervention z and a fixed decoding policy. The confidence branch reports c(q,z) before seeing any sampled answer. Correctness labels score answers; consistency alone is not correctness because repeated answers can be consistently wrong.

Process a fixed prefix such as “Subject: Ada Lovelace.” Apply the intervention at a verified subject-token location while processing that prefix. Clone the resulting key/value cache into independent answer and confidence branches. Both branches receive the relation question and fixed task suffixes. The confidence suffix asks for the chance that the specified answer policy will succeed. Neither branch receives the intervention label, sampled answer, or gold answer. Visible text stays identical across intervention conditions for each branch.

The suffixes can differ between answering and reporting, but their wording and decoding policies are frozen before evaluation. Record temperature, sampling parameters, token limits, answer scoring, confidence parsing, and invalid-output handling. Independent answer draws estimate success under that exact policy; eight draws are noisy observations, not an exact probability.

In Qwen, select an intermediate intervention site that changes later cached keys and values. A post-output hook on the final decoder block cannot carry its change into continuation through the cache alone. In the one-layer toy, use an embedding or pre-attention intervention instead.

Fit relation maps on training subjects. Gold labels may fit those maps and score outcomes; evaluation gold answers must not construct per-item perturbations or predictor features. Select layers, ranks, magnitudes, templates, and hyperparameters on development data. All relations and states of a subject stay in one split.

A concrete first intervention fits a regularized linear map from subject states to answer embeddings separately for each relation. Let P_r project onto the learned map's row space. Attenuate the centered subject state by h' = h − alpha P_r(h − mu), with mu estimated on training states. Select rank and alpha on development data, and match realized perturbation norm as well as rank in random controls. This is a candidate intervention whose causal selectivity must be measured, not assumed from decoding accuracy.

For each subject with relations r1 and r2, cross both queries with four states: clean, attenuation using the r1 map, attenuation using the r2 map, and matched random perturbation. This separates which relation was perturbed from which relation was asked. Conceal and randomize state assignment. The other-relation condition is already the cross-relation control, not an additional independent sample.

## 5. Experiments and decision gates

### S0 — Repair the text-only observer comparison

Question: how much of the saved probe advantage remains against correctly constructed text baselines?

Rebuild both tracks from the manifest and canonical collection used by the hidden probe. Validate subject splits, model/prompt/run identities, and intended draw counts: one answer per question for Track A and eight for Track B. Preserve legitimate repeated answers; do not deduplicate by answer text. Persist both tracks, provenance, observed AUROC differences, and subject-bootstrap intervals, including within-question intervals.

Fit question-only predictors for the prospective comparison. Add the candidate answer only for the after-answer comparison. Start with TF–IDF, relation, popularity, and length features, then test a development-tuned model that can represent question–answer compatibility. The current additive TF–IDF model is a baseline, not the strongest feasible observer.

Report overall AUROC and the eligible within-question subset separately. Cost: saved data and CPU for the first baseline. Completion means a reproducible comparison with matched provenance; it does not require a positive hidden-probe gap. A small gap weakens the historical motivation but does not invalidate an independently controlled private-state experiment.

### S1 — Validate the retained-state apparatus

Question: do the intended answer and confidence branches inherit the same manipulated state?

Preserve prefix masks and positions, or reject padded inputs until supported. Locate subject spans automatically. Compare forked and independent full runs for clean and intervened states, using the exact templates and declared dtype-specific numerical tolerances. Check branch independence, actual multi-token generation, answer scoring, and confidence parsing on the small model and then 7B.

Include a clean replay that reconstructs the same task from unmodified text. It should reset the intervention. Whether emitted confidence differs from retained-state confidence is a scientific outcome, not an apparatus requirement: native reporting failure must remain observable.

Cost: a small set of short continuations. Gate: cache equivalence, correct intervention location and retention, and a functioning prospective reporting interface. The earlier 0.5B smoke check does not by itself pass this gate.

### S2 — Debug interventions in a toy world

Question: can the intervention and analysis distinguish selective retrieval damage from generic disruption in a controlled setting?

Use 512 artificial entities, four random relation bijections, dimension 64 or 128, and three seeds. For a one-layer transformer, perturb subject embeddings or pre-attention inputs so cached keys and values change. A deeper toy can separately test routing interventions; label that as a different manipulation.

Compare clean, relevant-map, other-map, and rank/norm-matched random conditions. Test an external reliability probe and a reporting module separately. Train on one corruption family and evaluate on held-out entities and another family. If evaluating a fresh fact world, first establish factual competence in that world. For coherent fact swaps, define whether ground truth changes with the world; do not demand detection of a secretly false but internally indistinguishable replacement.

Cost: CPU or one small GPU. This is an apparatus and feasibility exercise. A toy-only result cannot establish a pretrained-model claim or replace the real-model selectivity gate.

### S3 — Run the 40-subject selectivity pilot

Question: does the relation-map intervention damage the queried relation more than the controls?

Use 40 development subjects, two relations, four states, and eight answer draws per cell: 2,560 short answer generations, plus confidence calls and extra controls. Estimate changes from clean separately for each queried relation. Compare relevant-map damage with other-map and random damage using paired subject-level uncertainty. Repeat random directions and shuffled-label maps across seeds on a small subset.

Choose useful damage and selectivity thresholds from development evidence, together with a tolerable amount of general disruption. Inspect ceilings, floors, and relation-map overlap. If selectivity fails, revise the intervention and repeat development work. Do not call generic corruption a selective retrieval manipulation.

Gate: a retained, selective, reproducible effect on real-model answering. This is required before M1-scale collection, regardless of toy performance.

### M1 — Test native confidence

Question: does the existing confidence channel predict reliability changes beyond the specified controls?

Reserve fresh evaluation subjects after design choices are frozen. An illustrative budget is 320–500 final evaluation subjects, excluding training, development, and calibration subjects. Two relations, four states, and eight draws require 20,480–32,000 answer generations before added controls. This is a budget estimate, not a power guarantee.

Use the primary endpoint in Section 6. Report native confidence, a separately calibrated version, external hidden-state probes, and baseline predictions distinctly. A native report can contain useful information while its numerical probabilities remain miscalibrated; measure both properties.

Decision: a native-report result supports C1 only when manipulation validity and the within-question/control checks also pass. If only an external probe succeeds, C1 has failed or is inconclusive, but there is reason to attempt M2. Probe success never establishes C2.

### M2 — Train reporting across states of the same question

Question: does pairing identical questions under different private states teach transferable reliability reporting?

Start with 500–1,000 training questions, four states per question, and 8–16 independent draws per state: 16,000–64,000 training answer generations. Freeze the answer model and enable the adapter only for reporting. Implement a training-capable reporting path; the current inference-only continuation is not a finished adapter trainer. Changing the answer policy later requires recollecting targets.

Match data, adapter capacity, and optimization budget across four arms: pointwise correctness supervision; out-of-fold hidden-probe distillation; pointwise supervision plus ranking across different questions; and pointwise supervision plus ranking across private states of the same question.

Use a proper scoring loss against sampled correctness. For ranking, account for uncertainty in state success differences; do not force hard orderings from nearly tied eight-draw estimates. Evaluate against independent draws. Add a no-private-state control: the same reporting adapter and labels, but reconstructed clean state. Compare independent rollout batches to show target precision; this is a sampling reference, not a guaranteed performance ceiling.

Hold out subjects and an entire intervention family. Untouched natural questions provide a separate non-degradation guardrail. Improvement there is a stronger deployment claim than transfer to unseen perturbations. One real-model seed is a feasibility result; use repeated seeds before a robust comparative claim.

Decision: C2 requires the learned report itself to succeed. A benefit specific to same-question pairing requires superiority over pointwise and random-pairing arms on frozen transfer evaluation. Training-family gains alone are insufficient.

### M3 — Test a causal pathway after a report succeeds

Question: can a candidate pathway be disrupted and restored while separating reporting sensitivity from factual answering and basic output ability?

Begin with 50–100 development pairs and a separate 50–100 evaluation pairs as a feasibility budget. Identify sites in development. Apply candidate manipulations upstream of the answer/report branch split, so both branches inherit them. Reporting-only, post-answer, and direct confidence-logit interventions cannot establish the intended dissociation.

Test ablation, donor patching, and restoration against random-site and rank/norm-matched controls. Measure answering, first-order uncertainty, ordinary number production, externally specified confidence reporting, and a second confidence format. Seek loss and recovery of reliability sensitivity with answer performance inside a predeclared equivalence margin.

Simulate power for that equivalence claim before collection. A nonsignificant accuracy change is insufficient; if the interval is too wide, report the pathway result as inconclusive. Even a selective pathway can implement first-order evidence readout, so C3 remains a functional mechanism claim.

## 6. Evaluation that matches the question

Separate selective damage from useful monitoring. A perfectly calibrated report has c = p regardless of which intervention caused the error. It should not need a larger confidence-versus-reliability slope for relevant than irrelevant damage. When an irrelevant intervention causes no damage, that slope is not well identified.

The manipulation test estimates paired changes in answer success for relevant, other-relation, and random perturbations. Report contrasts and subject-clustered intervals. Confidence-change contrasts are specificity diagnostics interpreted alongside actual success changes.

The primary monitoring endpoint is incremental held-out Brier improvement from adding the report to a prespecified baseline. Fit both predictors on training data and select/calibrate them on development data. Let b be the baseline and a the same predictor family augmented with confidence c. For independent correctness outcome y, compute:

Delta Brier = mean over subjects of [(b − y)² − (a − y)²].

Average draws, states, and relations equally within each subject before averaging subjects. Positive values favor the added report. Report the observed paired difference and a subject-bootstrap interval against a minimum useful improvement chosen before final evaluation. This tests additional information in the report. Separately report the Brier score of raw c to assess numerical calibration.

Use two different baseline information budgets. A text-only observer gets question, relation, and allowed metadata. The stronger diagnostic baseline additionally gets prespecified prospective answer-logit entropy/margin features and perturbation norm, but no gold answer, outcome samples, or intervention identity. Use improvement over this stronger diagnostic baseline as the primary comparison; the text-only comparison is secondary. The diagnostic baseline is privileged by design; do not call it text-only. Probe comparisons are also separate privileged methods.

Require evidence that the advantage comes from variation across private states, rather than only differences between questions. Evaluate report changes within question and compare with a preregistered within-question shuffle of report-to-state assignments. Recompute the full statistic in the shuffle analysis and retain subject clustering. Overall Brier improvement without evidence of state-sensitive reporting supports only the narrower predictive result.

Secondary metrics are AUROC, within-popularity-stratum discrimination, calibration plots, and report-format robustness. Report eligibility for mixed-outcome subsets. Do not interpret calibration on artificially balanced pairs as population calibration.

Use independent Bernoulli outcomes for scoring. Hierarchical success estimates can aid plots, but eight-draw fractions or shrunken estimates are not exact regressors. Any secondary slope analysis must account for binomial measurement error and shared-clean dependence.

Before freezing sample size, simulate the actual Brier endpoint and joint manipulation/monitoring decision using pilot estimates, subject clustering, two relations, shared clean states, ties, ceilings, and planned multiplicity handling. Choose sample counts from development data. Do not add draws to a promising final-test subset after inspecting its result; use a prespecified sequential design or a fresh confirmation set.

## 7. Interpretation of the controls

- Random directions and shuffled-label maps: match rank and strength and average over seeds. They can damage shared useful information. Test systematic relation specificity, not whether every random perturbation is harmless.
- Low-success questions: establish baseline success with independent draws. One wrong answer does not imply zero reliability. A confidence decline is appropriate if an already low success probability falls further.
- Entropy/margin matching: use as a secondary diagnostic and report overlap and discarded trials. Matching a post-intervention variable can induce selection effects and does not control every first-order computation.
- Clean replay: distinguishes access to retained state from fresh textual reconstruction. Confidence invariance can be a valid negative result.
- Odd wording: measure its effect on answer success before calling it harmless. It changes visible text and belongs in a separate analysis.
- Report format: repeat with another fixed elicitation format and explicit parsing rules, so one numerical prompt cannot carry the entire conclusion.

## 8. Immediate work and stopping rules

First repair S0 provenance and the S1 mask/template contract. Then debug the toy manipulation as needed and run the 40-subject real-model pilot. Use development evidence to freeze the intervention, endpoint, minimum useful effects, and sample size. Proceed to native reporting, learned reporting, and pathway tests only when their prerequisites hold.

If selective damage cannot be obtained, stop the relation-specific claim and revise the intervention. If probes succeed but reports fail, report readable information without claiming reporting success. If reports only recognize trained corruption patterns, restrict the conclusion accordingly. If precision is insufficient, report an inconclusive pilot rather than an absence of monitoring.

Existing implementation: `section33/analysis/text_baselines.py`, `scripts/run_text_baselines.py`, `section33/analysis/retained_state.py`, `tests/test_retained_state.py`, and `scripts/check_retained_state.py`. These require the corrections described above. Planned components, not completed artifacts: relation interventions, a retained-state generation collector, and a pilot runner. No new scientific experiment was run as part of this report consolidation.

## 9. References and evidence

[1] Ravfogel, Yehudai, Bruna, and Bietti. [Geometric Factual Recall in Transformers](https://arxiv.org/abs/2605.12426), 2026.

[2] Singh, Linzen, and Ravfogel. [Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242v2), 2026, version 2.

[3] Xi et al. [Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models](https://arxiv.org/abs/2609.33290), 2026.

[4] [Confidence Improves Self-Consistency in LLMs](https://arxiv.org/abs/2502.06233), 2025.

[5] Macar et al. [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396), 2026.

Repository evidence: `report/SECTION_3_3_CONFIRMATORY_REPORT.md`; `runs/section33_confirmatory/evaluation/`; and `runs/section33_confirmatory/predictions/test_predictions__qwen__qwen2.5_7b_instruct__trackB.parquet`. The archived review records the baseline-population, padding-mask, and one-layer-hook diagnostics. These references motivate the proposal; they do not establish its novelty exhaustively.
