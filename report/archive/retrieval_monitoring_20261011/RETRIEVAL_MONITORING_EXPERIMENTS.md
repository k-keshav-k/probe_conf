# Can a model tell when its factual recall is unreliable?

Small experiments inspired by Geometric Factual Recall and Can LLMs Introspect?

Research proposal • 11 October 2026 • Experiments below are proposed, not completed.

> **Update, 11 October 2026.** The revised plan is `report/RETRIEVAL_MONITORING_PLAN.md`.
> It applies the review fixes: separate native and learned claims, a paired-slope
> primary endpoint, extra controls, power gates, and two text-baseline regimes.
> Experiment 1 is complete for `runs/section33_confirmatory`. The strongest
> input-only observer is 0.065 AUROC below the hidden probe overall and 0.137
> below it within questions. The retained-state fork is implemented and tested
> (`section33/analysis/retained_state.py`, `tests/test_retained_state.py`,
> `scripts/check_retained_state.py`). A hook on the last decoder layer does not
> reach the branches; use a middle layer. The paths in this file are correct.
> This file stays as the original proposal.

## The proposed direction

Teach and test relation-specific monitoring of factual retrieval. Quietly weaken the model's internal access to one fact, keep the visible question unchanged, and test whether its confidence tracks the resulting change in its ability to answer. Require confidence to respond to damage relevant to the requested fact, rather than to any unusual internal state.

Example: weaken the internal information used to recall a person's birthplace. Confidence about birthplace should decline if recall becomes less reliable. Confidence about occupation should remain stable if occupation recall is preserved. An equally large but harmless perturbation should not produce the same confidence decline.

The candidate contribution is a controlled training and evaluation protocol for this behavior, followed by evidence about the pathway that produces the report. Better average calibration, an external probe, or detection of injected noise alone would not establish the claim. Treat novelty as provisional pending a broader related-work check.

## What the two papers contribute

Geometric Factual Recall in Transformers [1] separates stored relational information from the computation that selects the requested relation in a controlled transformer setting. Its real-model experiments find linearly decodable answer information in subject-token states; they do not establish that every pretrained LLM implements the toy selector. The useful inspiration is to intervene on relation-specific information and check effects on other relations, rather than search for another global MLP concentration statistic.

Can LLMs Introspect? A Reality Check [2] asks for privileged access beyond input cues and evidence distinguishing second-order monitoring from first-order computation. It shows why input-based prediction and generic anomaly detection can imitate introspection. Our proposal therefore needs identical-text private-state trials, strong external observers, harmless perturbation controls, and a later causal pathway experiment. Even a positive result would support a bounded functional monitoring claim, not settle the existence of strong introspection.

Nearby work already studies probe-to-confidence distillation [3], within-question confidence evaluation [4], and mechanisms of intervention detection [5]. The proposed distinction is whether confidence generalizes to the relation-specific consequences of unseen internal changes, beyond output uncertainty and generic perturbation signatures.

## Why this follows from the repository

The October 9 Section 3.3 report gives hidden-probe AUROC of about 0.888 before answering and 0.936 after answering, versus stated-confidence-probe AUROC of 0.592 and 0.571. The tested MLP additions provide negligible gains over the baseline. This motivates studying how an existing useful signal reaches a report, rather than assuming MLP summaries must outperform hidden states.

An exploratory calculation in this discussion used the saved Track B predictions. On the 341 of 4,000 test questions containing both correct and incorrect samples, mean within-question AUROC was 0.832 for the hidden probe, 0.737 for probability features, and 0.572 for the stated-confidence probe. This is after-answer discrimination; it neither establishes prospective monitoring nor excludes answer-content shortcuts. The old test data are now exploratory and must not become the new final holdout.

Implementation finding: section33/models/collect.py creates fresh confidence prompts through confidence_prompt_ids and calls generation again. An intervention applied only during the original answer pass would not automatically survive into that report. The new experiment needs a retained-state protocol and an explicit clean-replay control.

## Define the prediction target before running anything

For question q and concealed internal intervention z, the target is the probability that the answer branch produces a correct answer under a fixed decoding policy: p(q,z). Confidence c(q,z) is elicited before that answer is visible. Estimate performance with independent answer continuations from the same retained state; do not train on the intervention name or a corruption flag.

Use a fixed subject prefix followed by a relation question. For example: 'Subject: Ada Lovelace.' followed by either a birthplace or occupation question. Apply the intervention while processing the subject prefix, then retain the resulting state. Each condition gets the same visible tokens. Clone the state into an answer branch and a confidence branch, using fixed, predeclared suffixes for each task. Both branches inherit the altered subject representation; neither sees the experimenter's intervention label.

The branch suffixes differ between answering and reporting, but are fixed across intervention conditions. The estimand is the success probability of that exact answer branch. Confirm that state cloning, cache retention, and branch prompts preserve the intervention's behavioral effect. Do not reconstruct the reporting prefix from clean text and call it the same internal episode.

Natural-language confidence and an external hidden-state probe are separate methods. A probe can establish available information, but does not establish that the model uses it to report confidence. Benchmark simple output-margin and entropy predictors too: an internal advantage may still be ordinary first-order evidence.

## Experiment 1 — Check whether the existing probe gap needs an internal explanation

Question: Can a predictor that sees only the question and candidate answer explain the saved after-answer probe advantage?

Data and scale: use existing training/development labels and saved Track B test predictions. Preserve subject-disjoint splits and answer multiplicities. No new model training or generation is needed for the first baseline.

Procedure: fit regularized text-only predictors on question plus candidate answer, starting with word/character TF–IDF, answer length, relation, and question-side popularity. If locally available, add frozen input embeddings as a stronger input-only baseline. Tune only on development subjects. Compare with H, P, and raw/calibrated verbal confidence using mean within-question AUROC on the same eligible questions. Audit a small sample of labels and parse failures. Use subject-clustered paired intervals.

Interpretation: a text-only tie substantially weakens the interpretation that the original gap reflects special internal access. A surviving gap justifies the private-state experiment, but is not proof of privileged access: the external baseline may be weak. Report the strongest feasible observer and its compute.

Cost: CPU feature fitting over existing rows; optional embedding extraction is a separate small forward-pass job. Existing test results remain exploratory.

## Experiment 2 — Build a tiny world where storage and selection can be separated

Question: Can a confidence mechanism generalize across changes in which fact is stored, and distinguish relevant interference from irrelevant interference?

Data and model: start with 512 artificial entities, four random relation bijections over a shared attribute vocabulary, dimension 64 or 128, and a one-layer transformer. Use three seeds. First reproduce accurate recall and relation-specific linear decoding. If these checks fail, stop; the geometric interpretation is not established in this implementation.

Procedure: keep the answer model frozen. Fit relation readouts on one subject subset and validate on another. Construct clean, queried-relation attenuation, unqueried-relation attenuation, and norm-matched random conditions. Randomize interventions within each subject; cross the same altered subject representation with two relation queries. Fit an external reliability readout and a small in-model reporting module as separately labeled methods. Train monitoring on one corruption family and test on a different family, unseen subjects, and fresh bijections with appropriately reinitialized subject representations. Keep original answer-model memorization separate from monitor train/test splits: test facts must be stored even when monitor-training subjects are held out.

Controls: both relevant and irrelevant interventions must be present in monitor training, so 'any perturbation means low confidence' cannot solve it. Include a coherent whole-fact swap with updated world labels as a benign transfer control. If reference labels stay fixed after an undetectable coherent swap, the monitor may have no information that the stored fact is false; do not require impossible truth detection.

Readout: accuracy and Brier score across conditions; confidence response to relevant versus irrelevant damage; advantage over entropy, margin, perturbation norm, and input-only predictors. Give the norm-based predictor the norm as an intentionally strong diagnostic, while withholding intervention identity from the model.

Proceed if: the retained state contains a reliability signal beyond the simple controls and it transfers to new subjects/corruptions. If only the external probe works, information exists but the reporting mechanism has not learned to use it. If all gains vanish after uncertainty controls, retain the weaker first-order-confidence explanation.

Cost: tiny models and short artificial sequences; about 2,000 stored facts per world. This is the cheapest place to debug the causal design. It is an engineered mechanism test, not evidence about a pretrained LLM.

## Experiment 3 — Test relation-specific private changes in Qwen

Question: Does Qwen's confidence respond to the consequences of a hidden change in factual recall, beyond question familiarity and generic anomaly detection?

Data and scale: begin with 40 development subjects and 80 fresh evaluation subjects, two unambiguous relations per subject. Use separate training subjects to fit relation maps. Prefer relations with adequate baseline recall; a cohort pre-screened for baseline success is acceptable as a diagnostic if screening uses independent samples and the selected population is explicitly reported.

Geometry check: collect full subject-token states at a few development-selected middle layers. The saved prompt-end projections are not substitutes for these states. Fit small regularized relation readouts to answer representations using training labels. Use their row spaces as candidate intervention subspaces, not assumed memory locations. At evaluation, construct attenuation from the learned map and current state, without the test gold answer. Gold labels are allowed for training the map and judging outcomes, not as features supplied to confidence.

Interventions: identity/sham; attenuate information associated with the queried relation; attenuate an unqueried relation; random perturbation matched in norm and site. Pick layer, rank, and magnitude on development only. Require the targeted change to affect the intended recall more than unrelated recall. If it does not, the manipulation failed; no monitoring conclusion follows. Evaluate the same perturbation under both relation queries.

Reporting: run the retained-state confidence branch, the clean-prefix replay of the same confidence task, a hidden-state reliability probe, a text-only observer, and output-probability baselines. Never reveal whether an intervention occurred. A prompt-only anomaly condition is an additional control, analyzed separately because its visible text changes; confidence should follow measured success loss, not the mere presence of odd wording.

Measurement: four independent answer continuations per question/condition for the initial screen. Compare paired changes in correctness and confidence. Test whether confidence improves held-out Brier/log loss over text-only and uncertainty baselines. For an optional state-ranking analysis, define state ordering using a separate batch of rollouts, not the same outcomes used to evaluate it. Four draws give noisy per-state estimates; make aggregate paired comparisons and treat this as a feasibility pilot.

Cost: 120 subjects × 2 relations × 4 private conditions × 4 draws = 3,840 short answer generations, plus confidence branches and subject-state extraction. Prompt-only anomaly trials add about 960 answers if run at the same scale. No LoRA yet. These are workload counts, not measured GPU-hour estimates.

Proceed if: the intervention is selective and either the native report or a probe tracks the resulting reliability beyond controls. A probe-only positive result motivates training. Failure of native confidence is informative if the private-state probe succeeds. If selective manipulation fails, return to the toy study rather than interpreting it as lack of monitoring.

## Experiment 4 — Train confidence on the same question in different internal states

Question: Can training teach useful reporting of private retrieval reliability that transfers beyond the perturbations seen during training?

Data and scale: use 500–1,000 new training questions, with several private states per question. Start in the tiny world; run one small Qwen LoRA pilot only after Experiment 3 finds usable signal. Keep a new subject holdout. Hold out one intervention family entirely; freeze strengths and all evaluation choices before testing. Add untouched questions as a separate natural-distribution test.

Training arms: pointwise correctness supervision; hidden-probe distillation; pointwise supervision plus contrastive ranking across random questions; pointwise supervision plus ranking of different private states for the same question. Use the same examples, answer labels, optimization budget, and confidence interface. Use three seeds in the toy study; a single real-model seed is a feasibility result that needs replication.

Objective: combine a proper scoring loss on answer outcomes with a soft pairwise loss reflecting independently estimated reliability differences. Do not turn tiny, noisy differences from four samples into certain high/low labels. In training, use 8–16 outcomes per state, or shrinkage/uncertainty weighting. A normalized distribution over confidence bins makes the loss differentiable. Keep probe targets out-of-fold.

Start with the answer model frozen and activate the reporting adapter only in the confidence branch. This isolates learning to report; label it honestly as a learned reporting module. A later jointly active adapter requires recollecting outcomes because its own accuracy may change. Do not assume frozen teacher labels describe the updated model.

Success: same-question state pairing beats both ordinary supervision and random pairing on held-out subjects and an unseen corruption family; calibrated risk also improves on untouched questions. Evaluate confidence about the same answer policy, and verify any supposedly frozen answer behavior. Gains only on the trained perturbation pattern support perturbation recognition, not general retrieval monitoring.

Cost: four tiny-model arms first. Real-model costs are dominated by fresh supervision collection; 1,000 questions × 4 states × 8 draws is 32,000 short answers before adapter training. This is a gated follow-up, not the first experiment.

## Experiment 5 — Test whether a pathway carries retrieval reliability into the report

Question: Is there causal evidence beyond a confidence token directly responding to uncertainty or anomaly cues?

Data and scale: use 30–50 development pairs and 50–100 fresh evaluation pairs from successful earlier experiments. Begin in the controlled model. Real-model localization is a bounded follow-up with only a few development-selected sites.

Procedure: identify candidate intermediate states downstream of the perturbed subject representation and upstream of the report. Use paired donor/recipient states for the same question. Patch a small candidate subspace or connection; compare with random subspaces of matched rank and norm. Test ablation, donor transfer, and restoration. Measure the answer distribution in a separate branch with the same pre-branch manipulation, not after an answer has already been emitted.

Crucial control: changing confidence after freezing or printing the answer is not a meaningful dissociation; answer invariance would be automatic. Likewise, directly biasing the final confidence logits is only output control. A stronger result is that disrupting an upstream candidate pathway removes sensitivity to retrieval reliability while preserving recall, general language output, and simple externally cued confidence tasks; restoring the pathway restores the specific sensitivity. Repeat with another report format and unseen interventions.

Decision: seek a substantial loss of monitoring with an answer-accuracy interval inside a predeclared equivalence margin, rather than 'accuracy change was not significant.' With 50–100 pairs, a tight margin may be unidentifiable; call that inconclusive and plan a separate powered run. Record changes in answer entropy and other first-order summaries. A selective pathway supports a functional mechanistic account, but a first-order readout circuit can still mimic parts of this pattern. It does not conclusively establish second-order representation or consciousness.

Cost: a few thousand short patched forward/generation passes if the site grid stays small. Do not expand into a full circuit search before the earlier gates pass.

## Analysis rules and stopping decisions

Predeclare splits, manipulation strengths, primary contrasts, and a minimum useful effect using development data. These pilots estimate feasibility and variance; they are not powered confirmatory studies by default.

Cluster uncertainty by subject, keeping all relation, intervention, and sample rows together. Use paired contrasts and report multiplicity. For multiple primary comparisons use a declared correction or one primary endpoint. Layer and subspace selection belong entirely in development.

Use all answer outcomes for proper-score evaluation. Report eligibility and selection rates for any within-question or matched-uncertainty subset. Do not report calibration on an artificially balanced pair set as if it were population calibration. Matching on post-intervention entropy is a diagnostic with selection risks, not proof that all first-order information is controlled.

Separate three possible findings: an external probe can read reliability; the model's report uses private information; a causal mechanism specifically links retrieval reliability to the report. Each requires more evidence than the preceding finding. A useful negative result is that the probe succeeds but training or the native reporting channel fails.

Recommended order: Experiment 1 for a cheap reality check; Experiment 2 to validate the intervention and eliminate impossible targets; Experiment 3 for the first real-model test. Run Experiment 4 only if reliability is available internally, and Experiment 5 only if a report actually uses it.

The first defensible paper claim would be: 'Under controlled, relation-specific interventions, a learned confidence report predicts changes in factual-retrieval reliability beyond tested input-only, output-uncertainty, and generic anomaly baselines, with transfer to unseen interventions.' Strong introspection remains a separate, harder claim.

## References and repository evidence

[1] Shauli Ravfogel, Gilad Yehudai, Joan Bruna, Alberto Bietti. Geometric Factual Recall in Transformers (2026). https://arxiv.org/abs/2605.12426

[2] Shashwat Singh, Tal Linzen, Shauli Ravfogel. Can LLMs Introspect? A Reality Check (2026). Read v2, 21 August 2026, for the privileged-access and second-order distinctions. https://arxiv.org/abs/2605.26242v2

[3] Yadong Xi et al. Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models (2026). https://arxiv.org/abs/2609.33290

[4] Confidence Improves Self-Consistency in LLMs (2025). https://arxiv.org/abs/2502.06233

[5] Uzay Macar et al. Mechanisms of Introspective Awareness (2026). https://arxiv.org/abs/2603.21396

Repository sources: report/SECTION_3_3_CONFIRMATORY_REPORT.md; runs/section33_confirmatory/evaluation/decision__qwen__qwen2.5_7b_instruct__trackB.json; runs/section33_confirmatory/predictions/test_predictions__qwen__qwen2.5_7b_instruct__trackB.parquet; section33/models/collect.py; section33/models/loader.py. Existing helpers in section33/analysis/interventions.py address contribution perturbations and matching, but do not implement this retained-state, relation-specific protocol.
