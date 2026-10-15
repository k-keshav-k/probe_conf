# Can confidence track a model's own knowledge beyond shared difficulty?

Revised research report and next experiments • 12 October 2026

Status: the original S3 selectivity gate failed. Original M1–M3 remain on hold. Following meet.md, the priority is a novel test of model-specific competence beyond shared difficulty. The proposed next direction is a controlled competence crossover; donor patching is an optional later stress test. Neither direction passes a novelty gate merely by changing the intervention. No new model experiment was run for this revision.

## 1. Main direction: controlled competence crossovers

The theme in meet.md is whether confidence reflects the difficulty of a question or the particular model's ability to answer it. Preserve that distinction as the main scientific question. Treat the meeting's strong claims about architectural blindness and complete difficulty removal by bins as hypotheses, not established facts.

My recommendation is to change what the model genuinely knows while keeping the question and ground truth fixed. Make two copies of the same base model. Teach A a random subset of audited facts and B a complementary subset. Ask both exactly the same questions. Can their confidence distinguish “I learned this fact” from “this question looks familiar”?

Balance exposure to subjects, relations, answer vocabulary, training tokens, and optimization budget. The control must expose the same entities without revealing the tested entity–relation–answer binding; unpaired mentions must not assert false facts. Counterbalance assignments within subject and relation so neither checkpoint, relation, or popularity predicts which copy should succeed. Verify learning from answers rather than assuming training exposure equals competence.

The proposed novelty is a randomized competence crossover under matched observable difficulty and exposure, followed by a test of whether a reporting method follows new competence assignments. A generic accuracy–confidence correlation, before/after fine-tuning result, or perturbation detector would not be enough.

This is closer to the meeting's theme than making donor patching the headline. It also avoids requiring a model to detect an internally coherent false memory. The geometric paper motivates changing stored factual associations; it does not guarantee that LoRA produces a localized or storage-only change.

### N0 — Establish the novelty boundary before GPU work

Prepare a focused comparison against the closest methods:

- ESMA, Fine-Tuning Language Models to Know What They Know [3], already trains metacognitive alignment and evaluates newly acquired knowledge. “Confidence updates after learning” alone is not new.
- Looking Inward [4] already studies learned prediction of the model's own behavior. “Self-specific prediction” alone is not new.
- Probe-SD [5] already distills internal confidence into verbal reports. “Train a reporting adapter from a probe” alone is not new.
- Future Confidence Distillation [6] trains pre-solution predictors from post-solution probe estimates. “Predict later confidence before answering” alone is not new.

Compare their actual experiments with the proposed combination: randomized complementary knowledge assignments, matched exposure, identical evaluation questions, reversal of which checkpoint knows each fact, and transfer of the reporting method to unseen assignments. The focused search found close precedents; it did not establish that this combination is absent. Do not call it a first or commit a paper around it until this comparison is complete. If the distinction is already covered, abandon or sharpen this candidate before training.

The multi-model confidence data mentioned in meet.md could provide a cheap motivation analysis if available. Use leave-target-model-out difficulty estimates and question-controlled comparisons, with independent correctness draws. That analysis is supporting evidence, not the novelty claim; the full multi-model dataset was not located in this pass.

### N1 — Small competence-crossover feasibility test

After N0, start with roughly 80 audited subjects and two unambiguous facts per subject. Randomize which binding each copy receives, with both relations balanced across copies. Train two small knowledge adapters from the same checkpoint, initially on a 0.5B–3B model to check acquisition feasibility. Evaluate the base and both adapted copies on fixed, held-out paraphrases: 80 subjects × two relations × three checkpoints × eight draws = 3,840 short answers, plus prospective confidence reports and exposure-control diagnostics. Training cost is additional; no runtime estimate is claimed here.

Use a valid, common chat interface and an explicit answer policy. The confidence task receives neither the adapter name, training membership, gold answer, nor generated answer. Both answer and confidence calls use the same factual checkpoint. There is no hidden-state corruption in this experiment, so it does not depend on solving the failed S3 intervention.

First test actual competence reversal: A should outperform B on the facts assigned to A, and B should outperform A on those assigned to B. Check unrelated facts and reporting-format behavior for generic fine-tuning effects. Stop if the update does not produce the intended crossover; do not reinterpret mere exposure as a successful knowledge intervention.

Then compare the paired changes in confidence with independent answer outcomes. A question-only predictor gives the same prediction to both copies; a checkpoint-wide optimism shift cannot explain balanced opposite changes. Include prospective logit uncertainty and exposure/format controls to test those alternative explanations. Cluster by subject and randomization unit. Two checkpoints are a feasibility result; repeat independent assignments and training seeds before a general claim.

The facts are intentionally seen during knowledge acquisition. Held-out paraphrases test retrieval of acquired knowledge, not unseen-fact generalization. Later reporting evaluation must also hold out subjects, assignments, and checkpoint pairs from reporting supervision; call these different kinds of holdout explicitly.

### N2 — Train reports to follow competence, conditional on N1

Only if N1 produces clean competence crossovers, compare ordinary correctness supervision, probe distillation, and same-question/cross-checkpoint contrastive reporting. Use one shared reporting module across frozen knowledge checkpoints, without a checkpoint-ID input. Match data and optimization budgets, and use a proper scoring loss plus uncertainty-aware paired supervision.

Test on newly randomized knowledge assignments and entirely unseen subject groups, not merely another paraphrase of the reporting-training set. An adapter-label or memorized-training-membership heuristic should fail when assignment changes. Compare against matched text/exposure predictors and first-order uncertainty; also measure performance on untouched natural questions.

The candidate contribution is training confidence to follow experimentally reversed competence beyond the shared difficulty and exposure predictors. This supports a bounded functional self-assessment claim, not proof of second-order introspection. If only an external probe succeeds, report information availability; do not claim learned reporting until the reporting module itself succeeds.

Donor interference may later test transfer to another type of internal change. It is not required for the crossover pilot and is not an independently sufficient novelty claim.

## 2. What S3 established

The saved pilot has 40 subjects, 2,560 answer rows, and 320 confidence rows. It used Qwen2.5-7B-Instruct, layer 8, alpha 25. Positive damage means clean success minus intervened success.

- Relevant-map damage: +0.0172, interval [0.0031, 0.0328]; required mean at least 0.05.
- Other-map damage: +0.0266, interval [0.0094, 0.0469].
- Random damage: +0.0094, interval [−0.0125, 0.0313].
- Relevant minus random damage: +0.0078, interval [−0.0078, 0.0250]; required mean at least 0.03.
- Relevant minus other damage: −0.0094, interval [−0.0250, 0.0078].
- Clean success: 0.2375. Zero-success cell share: 0.5375.

Both registered mean thresholds fail. The other map causes more damage at the point estimate, but the paired interval includes zero; this does not establish statistically reliable reverse selectivity.

Recomputing the clean-success-at-least-0.25 subset gives 33 subject/relation cells, with damage 0.0417 relevant, 0.0606 other, and 0.0303 random. Restricting to these cells does not rescue selectivity. This is an exploratory subset chosen using noisy clean outcomes, not an independent confirmation or a proof that floors play no role.

The supported conclusion is specific: this tested relation-map attenuation procedure did not produce useful selective damage in this Qwen pilot. It does not show that relation-specific information is absent, that every intervention family will fail, or that the model cannot monitor its reliability.

## 3. Interpret the geometric diagnosis carefully

The supplied additional diagnostics report centered-state variance fractions of approximately 0.0000 for the uncentered-map projection and 0.0005 for centered and PLS variants, alongside held-out answer-embedding R² around 0.40. They also report stronger effects from all-subject-token donor replacement, with little benefit from all-token attenuation over random controls. These additional diagnostics were supplied in the discussion; their full artifacts were not located in the saved pilot directory.

These findings support abandoning this particular projection as the working manipulation. Low explained state variance alone, however, does not prove a no-op or identify the root cause. A low-variance direction can be highly predictive and have a large downstream gain.

For a full linear readout W and the orthogonal projector P onto its row space, WP = W. Removing P(h − mu) therefore removes the centered linear prediction W(h − mu), even if the removed component has small Euclidean norm. The implemented intervention instead keeps only the top few singular directions of W. Good performance of the full readout does not imply that this truncated subspace retains that performance.

Also, alpha = 25 is extrapolation, not gentle erasure. Along a projected component, h' = h − alpha P(h − mu) multiplies its centered value by 1 − alpha = −24. That interpretation is exact; whether the observed behavioral effect is an artifact is a separate empirical question.

Allow one CPU-only closure audit using saved states: compare centered full and truncated readout prediction quality, train-mean and shuffled-target baselines, projected versus remaining readout output, and equal-rank random variance capture. Check conditioning and solver precision. Archive the result; do not use it to justify another open-ended attenuation sweep.

This preserves the useful inspiration from Geometric Factual Recall [1]: distinguish stored information from the computation that selects and uses it. Linear decodability alone did not validate the chosen causal manipulation.

## 4. Fix the task before changing the intervention

There are two independent problems.

First, references and generation policy need auditing. The supplied examples include questionable occupation references. Saved clean answers also include long theological continuations for a father question about Jesus Christ. Such cases mix entity ambiguity, relation ambiguity, output length, and answer matching. A low measured success rate is not automatically a lack of stored knowledge.

Second, the two branches currently have different routes to subject identity. In collector.py, the answer branch receives a bare relation question, while confidence_suffix receives query["question"], which contains the original subject name. The latter can supply clean identity information after the prefix intervention. A parse rate of 100 percent does not validate that the report predicts the intervened answer policy.

Repair both branches around a single valid prompt format. Put the subject only in the shared prefix; use bare relation questions in both suffixes. Do not append a new complete chat conversation after a raw prefix without validating its semantics. The confidence branch must describe the exact short-answer task and sampling policy whose success it predicts.

Use manually audited, unambiguous facts with accepted aliases and an explicit short-answer policy. Avoid broad occupations, disputed parentage, and changing facts in the first pilot. Keep free generation as the main task; multiple choice would be a separately labeled task, since its candidates add information.

Preserve the existing mask and cache fixes. Recheck clean/intervened full-run versus fork equivalence for the revised templates, and inspect complete generated outputs rather than parser success alone. On diagnostic sibling branches, test whether the original subject can still be identified; those answers never enter the confidence branch.

## 5. Optional donor stress test — deferred

The following preserves the bounded donor design for possible later use. R0 and the task audit in R1 are useful now; R2–R3 are not the next GPU commitment. They do not replace the N0 novelty gate or N1 competence crossover.

### R0 — Close the old experiment

Save the failed S3 rows, settings, thresholds, and report unchanged. Record the negative finding and the limited geometric diagnosis above. Preserve the original gate rather than relabeling it as passed.

The repaired S0 artifact now records 2,000 training subjects and both tracks: question-only AUROC 0.789 on A; question-plus-candidate AUROC 0.866 on B; within-question B AUROC 0.663. The saved hidden probe remains stronger than these tested TF–IDF baselines. These are historical after-answer comparisons where applicable, not evidence that the new manipulation works.

The implementation README reports S1 passing on 0.5B and 7B and S2 passing on three toy seeds. Those checks were not rerun here. They do not cover the new branch templates or establish pretrained-model selectivity. The current package is retrieval_monitoring/, which supersedes the older implementation status in the October 11 plan.

Cost: saved-data inspection and the bounded CPU audit.

### R1 — Validate a clean task and reporting interface

Prepare roughly 20 development subjects with two audited relations each. Use eight independent answer draws per clean cell: 320 short generations, plus prospective confidence and a small identity/interface check.

Audit a stratified sample of generated answers manually, blinded to experimental condition when relevant. Separate semantic errors, missing aliases, truncation, and extra-answer formatting. Freeze scoring rules before the intervention comparison.

Choose a clear population with enough clean success to measure damage, using separate screening draws if selecting for headroom. Report exclusions and the resulting population; do not select or discard subjects by their later intervention effects. A practical development target is mean clean success around 0.60 or higher, not a universal validity threshold.

Gate: valid labels, a frozen short-answer policy, identical subject-information access across branches, working retained-state generation, and adequate headroom. If this fails, repair the task rather than patching more activations.

### R2 — Contrast interference with replacement

Use donor interpolation at all aligned subject-token positions:

h'_i = (1 − lambda) h_i + lambda h_i(donor).

Start with compatible subject-token lengths and positions; freeze an alignment rule before evaluation. Recompute downstream cache entries from the patched states. Same-subject replay is the exact sham. Validate that donor states come from matched templates and sites.

For each recipient, select audited donors with controlled facts. A useful conflicting donor differs on relation r1 but shares the answer to relation r2. A compatible donor shares the queried answers, providing a nonzero subject-change control. Match tokenization, coarse subject type, and state-distance distributions as far as feasible. Restrict to overlapping norm ranges and report matching failures.

Cross both relation queries with six conditions: clean; same-subject sham; compatible donor at lambda 0.5; conflicting donor at lambda 0.25; conflicting donor at lambda 0.5; and conflicting donor at lambda 1. Add matched random interventions on a small diagnostic subset.

Use a small development screen to choose at most two sites and freeze one before the pilot. Then run 40 fresh pilot subjects × two relations × six conditions × eight draws: 3,840 short answer generations and 480 confidence calls, excluding donor acquisition, screening, identity checks, and random controls.

Record original-subject accuracy, donor-answer frequency, answer diversity/consistency, prospective confidence, first-order uncertainty, and subject-identity checks. Score all outputs against the original visible question; do not silently change ground truth to the donor. Record donor-answer agreement separately.

Full replacement is a stress test, not a condition required to lower confidence. If every accessible cue supports the donor and no original-identity evidence remains, there may be no information from which any monitor could infer the experimenter's original truth. A confident wrong answer in that case does not by itself refute monitoring.

Donor matching controls behavioral consequences. It does not establish that only one relation's internal representation was changed. If the method succeeds, call it controlled retrieval interference or subject-representation replacement, not relation-specific erasure.

### R3 — Test reporting only on a validated manipulation

Before running R2, set a new development gate explicitly. A reasonable candidate is at least a 0.10 mean accuracy loss under partial interference and at least 0.05 more loss than the compatible-donor control, with reproducible direction across subjects. These are proposed engineering targets for a new manipulation, not revised pass criteria for old S3.

Require that original subject identity and general answer production remain usable in the partial-interference conditions. Set tolerances on R1 development data. Measure actual performance of the compatible donor; sharing reference answers does not guarantee a harmless intervention.

A 40-subject pilot is a feasibility screen. Estimate paired uncertainty by recipient subject; use disjoint donor/recipient groups or account for shared donors in clustering. If precision is insufficient, it is inconclusive. Freeze sample size and decision margins on development evidence before a separate confirmatory holdout.

For a valid manipulation, compare native confidence with a development-fitted baseline using question metadata, prospective output entropy/margin, and perturbation norm. The primary score is held-out incremental Brier improvement from adding confidence to that baseline, evaluated against independent Bernoulli answer draws and averaged within subject. Report raw-confidence calibration separately.

Inspect same-question changes and transfer across donor identities and strengths. Merely lowering confidence whenever lambda is nonzero is not enough. Do not claim all first-order explanations are excluded by a few entropy features. Keep answer stability as a separately labeled target: it must never replace correctness silently.

## 6. What the possible outcomes mean

- Partial interference reduces accuracy, preserves task identity, and confidence tracks the change beyond tested controls: proceed to a larger native-report test with the narrower retrieval-interference claim.
- A probe predicts these changes but native confidence does not: consider reporting-only adapter training; probe success alone is not a trained-report result.
- Confidence follows instability but stays high on coherent wrong replacements: a useful boundary between monitoring retrieval stability and monitoring truth. Verify this across donors before treating it as a general finding.
- Confidence falls for compatible and conflicting donors equally despite different accuracy effects: generic anomaly sensitivity remains a sufficient explanation.
- Every effective patch also replaces the subject or breaks the interface: no valid test of the intended retrieval-monitoring claim. Stop this manipulation family.

If donor-based reporting training is later pursued, it requires one of the first two donor outcomes. This is separate from N2's competence-crossover prerequisites. Compare pointwise correctness training, probe distillation, random-question pairing, and same-question/private-state pairing with matched budgets. Freeze the answer model, train only reporting, and test unseen subjects, donors, and a genuinely different intervention family before claiming cross-family transfer. A new lambda is a new strength, not a new intervention family.

Keep M3 pathway work deferred until a reporting channel actually succeeds. Any dissociation claim still requires upstream interventions shared by both branches, ablation and restoration, intact reporting ability, and powered equivalence of answer performance.

## 7. Priority and stopping decision

Immediate order: preserve S3 as failed; complete N0's nearest-work comparison; audit facts and the answer/confidence interface; then, only if the contribution remains distinct, run N1's small competence crossover. Defer donor experiments, broad model/site sweeps, reporting training, and pathway searches.

This is not a claim that knowledge updates are novel or guaranteed to work. The experiment must distinguish competence from matched exposure, question difficulty, and checkpoint-wide reporting shifts. If the nearest literature already makes that distinction, revise the research question before spending more compute.

If knowledge acquisition fails in the small model, report failed manipulation and stop that pilot rather than interpreting confidence. If acquisition succeeds but confidence does not follow, investigate whether internal states expose the difference before considering N2. If a learned report succeeds only on familiar checkpoint assignments, restrict the result to that setting.

Retain the original relation-specific negative result regardless of the new direction's outcome. A future positive finding must stand on its own protocol and controls. The first-order-versus-second-order distinction in Can LLMs Introspect? [2] continues to limit interpretation even when difficulty-based explanations are weakened.

## 8. Evidence and references

Local evidence read for this revision: runs/section33_v1/retrieval_monitoring/selectivity_report.json; pilot_rows.jsonl; pilot_pairs.csv; report/RETRIEVAL_MONITORING_S0_RESULT.json; retrieval_monitoring/collector.py; interventions.py; pilot.py; and README.md. The 33-cell headroom calculation was reproduced from saved rows. The projection-variance, centered/PLS, and donor diagnostics remain attributed to the supplied account until their artifacts are attached to the run.

The original consolidated plan and failed S3 report are preserved in report/archive/retrieval_monitoring_20261012_before_revision/. Earlier proposals and the detailed review remain in the October 11 archive. No experiment thresholds or saved pilot results were changed.

[1] Ravfogel, Yehudai, Bruna, and Bietti. [Geometric Factual Recall in Transformers](https://arxiv.org/html/2605.12426v1), 2026.

[2] Singh, Linzen, and Ravfogel. [Can LLMs Introspect? A Reality Check](https://arxiv.org/html/2605.26242v2), 2026.

[3] Park, Meyerson, Qiu, and Miikkulainen. [Fine-Tuning Language Models to Know What They Know](https://arxiv.org/html/2602.02605v1), 2026. Section 5.5 evaluates newly acquired information.

[4] [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/abs/2410.13787), 2024.

[5] Xi et al. [Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models](https://arxiv.org/abs/2609.33290), 2026.

[6] Kale. [Future Confidence Distillation in Large Language Models](https://arxiv.org/abs/2607.07626), 2026.

Theme source: meet.md. This revision treats novelty and alignment with its confidence-versus-difficulty question as explicit selection criteria, rather than equating a successful intervention with a novel research direction.
