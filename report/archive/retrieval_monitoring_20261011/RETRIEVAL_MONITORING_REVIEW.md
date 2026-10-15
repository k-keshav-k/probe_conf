# Review of the revised retrieval-monitoring plan

Reviewed 11 October 2026. Target: `report/RETRIEVAL_MONITORING_PLAN.md` and its supporting code/artifacts. The Google Doc still contains the original proposal; the local revised plan is the reviewed version.

Verdict: the direction remains worth pursuing, and separating native, learned, and causal claims improves the plan. It is ready for apparatus debugging after the fixes below, but not ready to freeze M1 as a confirmatory protocol. The most consequential issues are the primary estimand, the S0 evidence population, and the mismatch between the toy architecture and the intervention site.

This review did not change the plan or production code, refit probes, or launch model collection. It ran the existing seven retained-state tests and small CPU diagnostics.

## 1. High — The primary slope comparison can reject perfect monitoring

Plan lines 159–161 require the queried-relation slope of confidence change on reliability change to exceed the unqueried-relation slope. If confidence is perfectly calibrated, c(q,z) = p(q,z), both slopes equal one wherever they are identifiable. If unqueried attenuation has no reliability effect, its slope is poorly identified rather than necessarily small. The rule confuses selective damage with selective responsiveness per unit of damage.

Replace this with two distinct tests. First, establish selective recall damage through a paired contrast between queried and unqueried changes in success. Second, test whether confidence tracks the resulting changes and improves held-out prediction beyond the specified controls. Confidence should respond to equal reliability losses equally, even if one was caused by a nominally irrelevant perturbation.

A simple primary endpoint is paired Brier improvement over a development-fitted baseline containing question information, output-uncertainty summaries, and perturbation diagnostics. Score each prediction against the independent Bernoulli answer draws; average within subject. This avoids pretending the estimated success fractions are exact probabilities. Keep the queried-versus-unqueried confidence-change contrast as a specificity diagnostic. If retaining a slope, fit a joint model for the binomial counts and confidence rather than ordinary regression on shrunken probability estimates. Eight draws and beta-binomial shrinkage do not by themselves remove predictor measurement error; shared clean baselines also induce correlated changes.

## 2. High — S0 did not use only the stated frozen training cohort

Plan lines 47–74 describe frozen-train evaluation and declare Gate S0 passed. `load_answers()` at `section33/analysis/text_baselines.py:92` pools every matching combined/sharded label file without restricting to the manifest's population or enforcing canonical run provenance. Direct inspection found:

- The manifest has 2,000 training subjects.
- Both baseline loaders return 2,300 training subjects, including 300 outside the manifest.
- Track A has two training rows for 1,999 questions and one for 301.
- After multiplicity expansion, Track B has 16 rows for 1,843 questions and eight for 457; total 33,144 rows.
- There was no overlap between these training subjects and the current 4,000 test subjects.

This is a population/collection and weighting mismatch, not demonstrated train–test leakage. Its direction of effect is unknown. Rebuild the baseline from the same canonical collection and manifest used by H, validate split/run identities and per-question draw counts, then recompute the comparison. Do not deduplicate solely by candidate text: repeated identical answers can be legitimate draws.

The cited `text_baselines.json` currently contains only Track B reports, so it does not preserve the quoted Track A result. Save both tracks with provenance and the actual observed AUROC difference; the current overall delta field is a bootstrap mean. The within-question confidence interval is computed and discarded. Save it too.

Also narrow 'strongest feasible input-only observer' to 'tested TF–IDF baseline.' Its logistic score adds separate question and answer feature contributions, so within a fixed question it mostly ranks answer surface features, not question–answer compatibility. Fixed C=1 is not a development-selected observer. These baselines justify exploratory work, but do not exhaust strong input-only alternatives.

## 3. High — Several controls are assigned conclusions they cannot support

Plan lines 172–176 need weaker, outcome-conditioned expectations.

Floor effect: one wrong answer does not imply zero success probability. Damage may reduce an already low success probability or legitimately reduce confidence. A confidence drop on these items is not by itself evidence of generic anomaly detection. Use independently measured baseline performance and compare actual changes, with intervals; truly stable low-success conditions can serve as diagnostics.

Random map: a projection learned from shuffled labels can still remove useful shared information. The defensible expectation is no systematic relation specificity, after matching perturbation strength/rank and using several random seeds—not guaranteed absence of any recall or confidence change.

Clean replay: native confidence can fail to use a private state. A difference in emitted confidence must therefore be an experimental outcome, not an apparatus pass criterion. Apparatus validation should establish that the intervention is retained and changes relevant computation; reporting invariance can then be a valid negative scientific result.

Matching entropy or margin conditions on a post-intervention quantity. Keep it as a diagnostic and report discarded trials/overlap. It does not establish independence from every first-order computation.

## 4. High — The one-layer toy cannot use the current post-block hook

Plan lines 103–107 correctly state that a post-output hook on the final decoder block cannot affect later continuation through the KV cache. Lines 125–126 then specify a one-layer toy model using the same protocol. Its only decoder block is also its final block.

A tiny one-layer Qwen reproduction gave exactly zero change in branch logits after this hook. Use an embedding or pre-attention intervention that changes cached K/V for the one-layer storage experiment, or explicitly use a multi-layer toy for post-block interventions. These are different experimental manipulations and should be labeled accordingly. Retaining `last_hidden` does not fix this because the current branch function does not consume it.

## 5. High before collection — Passing the fork tests does not yet validate the experimental protocol

All seven existing tests pass. They establish useful single-example cache behavior. Two remaining boundaries matter:

- `continue_branch()` at `section33/analysis/retained_state.py:138` replaces the retained attention mask with all ones. A padded-prefix CPU reproduction differed from the correctly masked full run by 0.0351 in maximum logit value. Preserve the prefix mask and append valid suffix masks, or explicitly reject padded inputs until supported. Also define the subject-token position for each batch item.
- `scripts/check_retained_state.py:49–55` uses a complete QA prompt followed by teacher-forced strings ' Paris' and ' 85'. This tests continuation after supplied tokens, not the intended subject-prefix / relation-question / prospective-confidence interface or actual free generation. The default token offset is not verified to be a subject token.

Before M1, test the exact proposed templates, automatically located subject spans, clean and intervened fork-versus-full equivalence, actual multi-token generation, and the sampling/parse policy on 7B. A logit change in an arbitrary suffix is not yet evidence that native confidence about the specified answer policy was elicited. Keep the current status as 'cache primitive tested; experimental integration pending.'

## 6. High before freezing M1 — Power and follow-up sampling do not match the endpoint

The table at plan lines 80–93 is a normal-approximation paired mean-change calculation, not power for the slope difference, a noisy latent probability model, or the joint selectivity-and-monitoring gate. Larger N alone does not repair that mismatch. Simulate the corrected endpoint with subject clustering, two relations, the shared clean condition, realistic success rates, eight Bernoulli draws, confidence ties/ceiling effects, and any multiplicity correction. Clarify whether 320–500 refers to final test subjects or the entire fresh cohort.

At lines 151–152, collecting 16 draws on a smaller 'confirmatory subset' when the effect is close to a gate makes follow-up sampling depend on observed results. Use a development-only rule followed by a fresh fixed-size holdout, or a prespecified sequential design. Revisiting a subset of the same test set is not an independent confirmation.

At 320–500 evaluation subjects, two relations, four conditions, and eight draws, the core design already requires 20,480–32,000 answer generations, before extra controls and confidence prompts. Keep the 40-subject selectivity dry run as the immediate task.

## 7. Medium — The evidence gates overstate what has been achieved

Plan lines 182–184 say a probe-only positive result supports C2 with a learned module. It only motivates testing C2: that claim requires the learned reporting module itself to succeed in M2. Retain three distinct states: information readable by an external probe; native reporting succeeds; trained reporting succeeds.

Make the stage order explicit. The opening says every earlier gate must pass; S1 failure sends work to S2; S2 is then described as a prerequisite but also as motivating M2 directly. Prefer: repair S0 provenance; validate the cache; debug interventions in the toy as needed; pass the real-model selectivity pilot; then freeze M1. A toy result cannot substitute for real-model manipulation validity.

For M2, define the no-state and noise-ceiling arms precisely and restore the proper-scoring/ranking objective and uncertainty treatment from the original proposal. Improvement on untouched questions is useful, but distinguish that stronger deployment claim from transfer to unseen private perturbations; non-degradation on untouched questions may be an appropriate guardrail for the narrower claim.

## 8. Medium — Restore the safeguards that made C3 meaningful

The shortened M3 section no longer explicitly requires the same candidate-pathway manipulation to affect both branches before their divergence. Without that, confidence can change while the answer stays fixed simply because the intervention occurs only in the reporting branch. Exclude direct confidence-logit manipulation and post-answer interventions as evidence of dissociation.

Require an upstream intervention, random-site/rank controls, loss and restoration of reliability sensitivity, preservation of recall within a powered equivalence margin, and checks that general number/report production remains intact. Use a second reporting format. Even this supports a bounded functional pathway claim; it does not by itself establish second-order representation.

## Recommended immediate action

Repair the S0 data selection and the cache-mask contract; replace the primary slope comparison; correct the control interpretations; and choose a valid intervention site for the toy model. Then run the 40-subject selectivity and real-template integration pilot. Do not spend on M1-scale collection or LoRA until those checks pass.

Minor editorial fixes: the P3 reference should point to section 4.3, P5 to section 3.2, and the summary's 'exact match' should read 'within declared numerical tolerance.'
