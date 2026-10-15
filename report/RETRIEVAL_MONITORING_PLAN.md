# Can training selective retrieval improve metacognitive confidence?

Revised research report and small experiments • 12 October 2026

Status: proposed, not implemented or run. The original S3 attenuation gate failed; original M1–M3 remain on hold. The new priority is to train a retrieval mechanism and its confidence report, then test whether the report tracks the model's own competence beyond shared question difficulty. This is a new hypothesis, not a reinterpretation of the failed pilot.

## 1. Recommended direction: train selection and monitor its reliability

The proposal is simple: help the MLP select information relevant to the question, and teach the model to recognize when that selection succeeds or fails. Ask whether improving the selection mechanism makes useful confidence easier to learn. This directly follows meet.md's interest in instilling metacognition through internal computation.

Three claims must be separated. More concentrated channel activity is sparsity. Retrieving the requested attribute without interference from other attributes is functional selectivity. Knowing whether the resulting answer is correct is metacognitive reliability. None automatically implies the next.

Geometric Factual Recall [1] supplies a mechanism to investigate: its controlled models learn a relation-conditioned selector over structured attribute representations, using ordinary language-model training. Its real-model experiments establish linear decodability, not the same causal selector in Qwen. Our extension is a hypothesis about training that mechanism and its reliability report, not a result already demonstrated by the paper.

The frozen-model MLP results do not close this direction. The confirmatory study found essentially no useful incremental prediction from the tested MLP summaries beyond the baseline containing hidden states: AUROC gains were +0.0038 before answering and approximately zero after answering. That limits the case for extracting more summaries from an unchanged model. It does not establish that changing the MLP cannot improve retrieval or make reliability easier to report.

The candidate contribution is: training functionally selective retrieval improves transferable, model-specific confidence, with interventions showing that the learned reporting benefit depends on the trained computation. Better answer accuracy alone, lower activation entropy alone, or another proper-score RL method would not establish this contribution. Novelty remains provisional.

### T0 — Define what is being trained

For Qwen's normalized MLP input x, the actual channel coefficient is s_j = SiLU((W_gate x)_j) × (W_up x)_j. Channel j contributes u_j = s_j W_down[:,j]. Contribution mass is ||u_j||. A raw dot product x · W_down[:,j] is neither that coefficient nor a measure of correctness.

Initially train small adapters on W_gate and W_up in one or two development-selected layers; freeze W_down, attention, and embeddings. This restricts the update toward channel selection but does not prove it changes only routing. Include a capacity-matched ordinary adapter control. If fixed upstream representations do not contain usable information, this restriction may fail.

Measure selection after both the subject and requested relation have entered the causal context, initially at the final question token. A subject position preceding the relation cannot condition on that later relation. It can be a storage diagnostic, but not evidence of relation-conditioned selection.

Use mass concentration and signed alignment as diagnostics. Do not maximize them as the main reward. A concentrated, strong association can be wrong [10]. Training should reward correct use of relevant information and resistance to irrelevant information, with correctness established independently.

### T1 — First experiment: a controlled selection task

Start with a small one-layer transformer, approximately 512 subjects and four relations sharing an answer vocabulary, across three seeds. Construct a world with known attribute components and freeze its encoding while training the selector. This gives exact interventions without fitting the failed Qwen relation maps.

Use clean inputs, inputs with an irrelevant attribute changed, and inputs with the requested attribute missing or partially obscured. Include both harmless and harmful changes at matched strengths, and enough variation in available information to avoid ceiling performance. Do not demand uncertainty on an internally coherent wrong memory that contains no evidence of its falsity.

For a coherent attribute swap, define explicitly whether the evaluation world also changes. If it does, the new answer is correct in that counterfactual world and confidence need not fall. If ground truth stays fixed, the swap is a corruption, not a new truthful training example.

The selectivity objective should promote correct retrieval and invariance to irrelevant changes. It must not reward making the model vulnerable to target removal. Use target removal as an evaluation of dependence on relevant information, with confidence judged against measured success rather than a prescribed low value.

Freeze the learned selector and test fresh attribute assignments encoded with the same construction. Also hold out intervention strengths and subjects from confidence supervision. This checks transfer beyond memorizing factual bindings. The constructed world is a mechanism feasibility test, not evidence that pretrained LLMs use this representation.

### T2 — Optimization: RL plus a direct selectivity loss

Alternate two updates so the prediction target stays clear.

- Retrieval update: optimize sampled-answer correctness with RL, plus a differentiable selectivity loss on controlled examples. The auxiliary loss can combine answer cross-entropy on valid counterfactual worlds and output-distribution consistency under irrelevant changes. Freeze the reference branch for consistency; retain the correctness anchor to prevent a constant-output solution.
- Reporting update: freeze the current answer policy and train verbal confidence c with reward −(c − y)^2, where y is the correctness of an independently sampled answer from that policy. Generate that answer on a sibling branch that never reads c. Refresh answer outcomes after retrieval updates. No gold answer or generated answer enters the prospective confidence branch.

For a fixed answer policy, expected squared-error reward is maximized by reporting its probability of success, subject to the reporting model's capacity. It is not a guarantee of calibration after distribution shift. Keep reporting updates out of the answer policy in this first experiment so confidence training cannot manufacture an easier target.

A pre-answer geometric statistic is identical across completions of the same prompt. Added as a detached GRPO reward, it cancels under group centering; more generally its action-independent score-function contribution has zero expectation. To directly reshape those activations, backpropagate an auxiliary loss. Calling a detached mass bonus “RL on the MLP” does not solve this credit-assignment problem.

An alternative is a stochastic routing controller whose sampled channel masks are actions, rewarded for correctness and calibrated confidence. Then policy gradients can credit routing choices directly. This changes the architecture and introduces inference randomness; defer it unless the simpler hybrid establishes a benefit.

RL is not assumed superior. Compare the hybrid against supervised answer/confidence training with the same selectivity loss and data. If supervised training matches it, the result is about training selective retrieval, not a special advantage of RL.

### T3 — Small comparison that can reject the idea

Use the following arms with matched data and explicit parameter/update budgets:

- Frozen base with confidence training only.
- Answer-correctness RL plus confidence training.
- The same RL and confidence training plus the selectivity loss.
- Supervised answer/confidence training plus the same selectivity loss.

Keep an untouched base as a reference. The essential comparison is the third arm against the second: does selective-retrieval training add anything beyond ordinary task and confidence training? The fourth checks whether RL is needed. Log actual rollout and optimization costs rather than labeling unequal budgets “matched.”

Measure answer accuracy; functional selectivity under held-out interventions; prospective Brier score; within-difficulty-bin discrimination; and same-question changes in confidence versus independently measured success. Use subject-clustered intervals and report seeds. Bins control observed difficulty covariates, not every source of difficulty.

Retrain equally expressive hidden-state and MLP-feature probes on each trained model. This distinguishes a change in information available to an observer from a change in the native reporting channel. If an extra feature observer is needed, use the same feature-injection interface for a capacity-matched hidden-state observer and label that as an architectural extension.

Run a sham edit and matched random intervention alongside target and irrelevant interventions. After establishing the relevant computation, attenuate and restore it on held-out examples. Check whether the reporting benefit changes with the computation while unrelated language/reporting ability remains intact. A generic intervention detector is an alternative explanation until these controls succeed.

Stop the mechanism claim if concentration rises without functional selectivity. Stop the metacognition claim if selectivity improves but confidence does not outperform the ordinary RL arm. If accuracy changes substantially, stratify and report the remaining errors, use a proper-score decomposition, and include same-question conditions; lower aggregate Brier alone may follow from easier prediction.

### T4 — Small pretrained-model transfer, only after the toy result

Start with one 0.5B–3B model and approximately 400 audited subjects, two unambiguous relations each, split by subject into 240 train, 80 development, and 80 test. This is a feasibility screen with limited statistical precision. Use rank-8 gate/up adapters in at most two layers, selected only on development data. Freeze layer choices, margins, and compute before a larger confirmation.

Initially test the two decisive arms: ordinary answer/confidence RL and the same training with a validated selectivity loss. Eight independent short-answer evaluation draws per question would be 6,400 answers per trained model across all 800 questions; training rollouts and repeated seeds are additional. No wall-time estimate is claimed.

Do not transplant exact toy interventions into Qwen by assumption. First demonstrate that a proposed training perturbation preserves task identity and changes relevant versus irrelevant information as intended. If this fails, the real-model selectivity experiment is blocked; output-level relation discrimination alone supports a weaker behavioral claim.

Include high-consistency wrong answers in evaluation, not just obscure questions. Stability is not truth. Hold out subjects and paraphrases, then test a new relation or dataset only after feasibility. Keep clean factual accuracy and ordinary text performance as constraints.

### T5 — Main-theme evaluation: confidence follows private competence

Use complementary factual updates as an evaluation of the trained reporter: two copies receive different bindings while subject exposure, relation frequency, answer vocabulary, and training budget are balanced. Both answer identical questions. Freeze the reporter and test new assignments with no adapter ID supplied to it.

Measure actual competence reversal from answers; exposure is not competence. Facts seen during knowledge acquisition are not unseen facts. Hold out reporting-supervision subjects and checkpoint assignments separately from retrieval paraphrases.

This is not the novelty by itself. Kadavath et al. [8] already compare self-knowledge targets across differently trained models to separate self-knowledge from general difficulty. Our proposed question is whether selectively trained retrieval improves reporting transfer under controlled competence changes. A question-only predictor and checkpoint-wide optimism shift are necessary but insufficient controls; include logits, text/exposure predictors, and hidden-state reporting baselines.

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

## 5. What the requested papers change

- SelfAware [7] studies recognition of answerable versus unanswerable questions and improvements through prompting/instruction tuning. Our target must be the particular model's success probability, not only whether a question is answerable in principle.
- Kadavath et al. [8] train P(IK) using sampled correctness, including full-model updates, and investigate model-specific versus shared-difficulty targets. Heads, sampled-success supervision, and self-versus-other comparisons are established starting points.
- CORE-PO [9] uses confidence in reasoning and answers to construct preference-training signals. Its accuracy improvements motivate training from confidence, but do not establish that increasing confidence yields calibrated self-assessment.
- Cheang et al. [10] find that subject-grounded hallucinations can resemble correct factual recall in the tested internal signals. This is why selective or strong activation cannot serve as a truth label.
- Reported Confidence Tracks Commitment [11] motivates separating confidence from willingness to answer. As a secondary control, vary abstention stakes while keeping the predicted answer policy fixed. Do not require invariant confidence if stakes change that policy.
- RLMF [12] already uses GRPO, consistency-based intrinsic-confidence estimates, factual calibration, and metacognitive feedback. Generic confidence RL or distilling consistency is therefore not the new contribution here.

Also compare ESMA [3], learned behavior prediction [4], probe distillation [5], and future-confidence distillation [6]. The proposed distinction is training a causal retrieval mechanism and testing whether that produces transferable native confidence beyond difficulty. This focused comparison is not proof that no prior method does it.

## 6. Priority, gates, and claim boundaries

Immediate order: audit answer labels and interfaces; run the toy mechanism test and four-arm comparison; proceed to a small real-model experiment only if functional selectivity and reporting benefits both appear. Set numeric practical-effect margins using development precision before opening test results. A small ambiguous result warrants an uncertainty report, not relaxed gates.

A successful result needs all three: selective retrieval on held-out causal tests; confidence gains over ordinary answer/confidence training; and evidence beyond question difficulty or generic anomaly sensitivity. An external probe gain alone is not native metacognition. An accuracy gain alone is not a reporting gain. Do not claim privileged introspective access from these functional tests [2].

Keep donor patching as an optional later stress test. It changes subject-level content and can install coherent false associations. It is not a substitute for relation-selective training or an independently novel contribution. The earlier donor protocol remains in the archived plan.

The original S3 gate and its failed verdict remain unchanged. Original M1–M3 are not automatically reopened by this new program. The new experiments need their own preregistered comparisons and validated intervention, rather than inheriting the failed attenuation procedure.

## 7. Evidence and references

Local evidence: report/SECTION_3_3_CONFIRMATORY_REPORT.md; section33/features/mlp_math.py; runs/section33_v1/retrieval_monitoring/selectivity_report.json; pilot_rows.jsonl; pilot_pairs.csv; report/RETRIEVAL_MONITORING_S0_RESULT.json; retrieval_monitoring/collector.py; interventions.py; pilot.py; and README.md. The 33-cell headroom calculation was reproduced from saved rows. Additional projection-variance, centered/PLS, and donor diagnostics remain attributed to the supplied account until saved artifacts are available.

The original consolidated plan and failed S3 report are preserved in report/archive/retrieval_monitoring_20261012_before_revision/. The intermediate competence-crossover proposal is preserved there as COMPETENCE_CROSSOVER_REVISION.md. No model experiment, source-code change, or pilot-threshold change was made in this revision.

[1] Ravfogel et al. [Geometric Factual Recall in Transformers](https://arxiv.org/html/2605.12426v1), 2026.

[2] Singh, Linzen, and Ravfogel. [Can LLMs Introspect? A Reality Check](https://arxiv.org/html/2605.26242v2), 2026.

[3] Park et al. [Fine-Tuning Language Models to Know What They Know](https://arxiv.org/html/2602.02605v1), 2026.

[4] [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/abs/2410.13787), 2024.

[5] Xi et al. [Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models](https://arxiv.org/abs/2609.33290), 2026.

[6] Kale. [Future Confidence Distillation in Large Language Models](https://arxiv.org/abs/2607.07626), 2026.

[7] Yin et al. [Do Large Language Models Know What They Don't Know?](https://arxiv.org/html/2305.18153v2), 2023.

[8] Kadavath et al. [Language Models (Mostly) Know What They Know](https://arxiv.org/html/2207.05221v4), 2022.

[9] Jang et al. [Self-Training Large Language Models with Confident Reasoning](https://aclanthology.org/2025.findings-emnlp.806.pdf), 2025.

[10] Cheang et al. [Large Language Models Do NOT Really Know What They Don't Know](https://arxiv.org/html/2510.09033v1), 2025.

[11] [Reported Confidence in LLMs Tracks Commitment More Than Correctness](https://arxiv.org/html/2606.29490v1), 2026.

[12] Liu et al. [Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs](https://arxiv.org/html/2606.32032v1), 2026.

Theme source: meet.md. Mechanistic and architectural claims in the meeting remain hypotheses to test.
