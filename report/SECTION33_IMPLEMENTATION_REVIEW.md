# Review of the section33 implementation

Date: 8 October 2026.

**Verdict: partial implementation. It is not ready to give a definitive answer to section 3.3.**

The package contains useful feature extraction and statistical components. However, errors in feature selection, information boundaries, data joins, and power estimation change the experiment being tested. Several required controls exist only as helper functions. The current saved experiment is a small exploratory run, not the planned confirmatory experiment.

This review compares `section33/`, its tests, and `runs/section33_v1/` with `meet.md` section 3.3, the corresponding text in `meet.original.txt`, and `report/SECTION_3_3_TEST_PLAN.md`. No model collection, probe refitting on the real cohort, or production changes were performed.

## 1. What the saved results establish

The cohort manifest records 600 subjects: 300 training, 150 development, 100 calibration, and 50 test. The source is a PopQA export. There are three relations; mother is absent. The requested test size was 183, but the pipeline accepted 50. The plan instead calls for a fresh four-relation cohort, with an initial 2,400 test subjects and a development-based power check.

Only a Qwen Track B decision file is present. Track A collection files exist, but a primary Track A decision is absent. There is no saved Mistral replication decision, power table, or intervention outcome file.

The saved Track B decision reports:

| Quantity | Saved estimate | Saved 95% interval |
| --- | --- | --- |
| AUROC gain, B+M minus B | +0.0384 | −0.0060 to +0.0887 |
| Brier improvement | +0.0093 | +0.0001 to +0.0201 |
| Increase in corrected calibration error | — | −0.0194 to +0.0146 |
| Within-question AUROC gain | +0.0051 | −0.0317 to +0.0417 |

Only 9 of the 50 questions have both correct and incorrect sampled answers. The saved verdict is **inconclusive** for both comparisons. Its AUROC interval includes zero, its Brier lower bound does not exceed the planned +0.005 margin, and its calibration guardrail does not pass. The within-question interval crosses the +0.030 margin.

These numbers describe the existing output. They must not be treated as valid estimates for the intended experiment until the errors below are corrected. In particular, the named B baseline lacks probability features, and four sampled answers are absent from the joined evaluation data.

Sources: `runs/section33_v1/manifests/cohort_manifest.json` and `runs/section33_v1/evaluation/decision__qwen__qwen2.5_7b_instruct__trackB.json`.

## 2. Findings that block a confirmatory answer

### F1 — High: the probability baseline selects zero features

Location: `section33/probes/pipeline.py:47`, `section33/features/probability.py:24`, and `section33/probes/pipeline.py:198`.

`Block.matches()` requires `__last_prompt__` or `__answer_last_content__`. Probability feature names end at the position, without the final `__`. Consequently, neither P block selects any probability columns. An empty feature matrix silently produces a constant prediction.

**Verified:** the saved feature rows contain probability features for both tracks, but the selectors return zero columns. The saved calibrated P score has one value, approximately 0.076585, for every test candidate; its AUROC is 0.5.

**Effect:** B is not D+P+H. The experiment cannot test whether MLP features improve confidence beyond the intended probability baseline. Comparisons against P are also invalid.

**Required correction:** fix position matching; reject empty required blocks; test the collector-to-selector connection. Refit and recalibrate the affected methods after correction.

### F2 — High: Track A uses information from the generated answer

Location: `section33/models/collect.py:393`, `section33/datasets.py:105`, and `section33/cli.py:222`.

Track A activation extraction correctly uses only the prompt. However, its feature row also stores `content_length=len(answer_ids)`. `metadata_features()` includes this value in D for both tracks. B and B+M therefore receive answer length after generation.

**Verified:** changing only answer length from 1 to 7 changes the Track A baseline design matrix.

**Effect:** this is future-answer leakage relative to the before-answer task. It does not prove reference-answer leakage, but it invalidates the claimed prediction time.

**Required correction:** use a track-specific metadata allowlist. Track A must use only information available before the answer. Add a test that changing generated-answer fields leaves every Track A predictor input unchanged.

### F3 — High: candidate deduplication drops sampled answers

Location: `section33/datasets.py:62` and `section33/datasets.py:77`.

Track B labels are deduplicated by decoded candidate text. Features are deduplicated by token IDs. The subsequent join still uses `answer_index`, despite the docstring describing a candidate-based join. Different token sequences can decode to the same text. The label deduplication then discards one row without preserving its multiplicity. Batch-local answer indices also make repeated collections fragile.

**Verified:** a two-candidate example with the same text, different token IDs, and total multiplicity 8 returns multiplicity 4 after joining. In the real saved test, labels contain 207 candidate rows with total multiplicity 400. Predictions contain 205 rows with total multiplicity 396. Two questions lose samples.

**Effect:** the evaluation no longer includes all eight draws per question. Candidate probabilities and equal question weighting change.

**Required correction:** use one stable candidate identity across all tables, including token identity and run/model provenance. Validate join cardinality and reject unmatched rows. Assert exactly eight retained draws per completed question. Do not combine distinct token sequences merely because their decoded text matches.

### F4 — High: training and calibration ignore sampling multiplicity

Location: `section33/cli.py:270`, `section33/cli.py:345`, and `section33/probes/calibration.py:50`.

Probe fitting, inner validation, development scoring, and calibration use one row per unique candidate. Only final evaluation expands multiplicity at `cli.py:371`.

**Effect:** a candidate sampled seven times and a candidate sampled once receive equal weight during fitting. Questions with more unique candidates also receive more weight. The fitted probabilities target a different distribution from the eight-sample evaluation population. Even the per-method AUROC and headline paired AUROC use different weighting in the saved report.

**Required correction:** apply multiplicity consistently to training, validation, calibration, development, and evaluation, while keeping subject groups intact. Expansion or equivalent sample weights can work. Verify equivalence against an explicitly expanded eight-draw dataset, including preprocessing and calibration.

### F5 — High: the power analysis cannot size the intended experiment

Location: `section33/cli.py:232`, `section33/cli.py:297`, and `section33/analysis/power.py:34`.

There are two separate errors:

1. `_subject_deltas()` calculates AUROC inside each subject and discards subjects with one label class. The planned Track A has one answer per subject, so every subject is discarded. Development then skips the power table. A pooled AUROC difference cannot be replaced by the mean of these within-subject differences.
2. The simulator draws estimated effects with covariance equal to the subject-level covariance, rather than a sampling covariance that decreases with N. Only its interval width decreases with N. The calibration guardrail is represented by an assumed random pass probability, not simulated predictions.

**Verified:** four one-answer subjects return two empty delta arrays. With a true assumed gain of +0.040 AUROC and +0.010 Brier, and calibration forced to pass, the simulator reports only 44.17% useful-gain detection at one million subjects. It still reports 15.83% harm. This follows from the variance-scaling error.

**Required correction:** simulate paired predictions and outcomes for the actual estimands, preserving subjects, prevalence, strata, and calibration. Include separate within-question power. Freeze sample size from development and enforce it before final evaluation. Fixing `/N` alone does not repair the wrong AUROC estimand.

### F6 — High: the final comparisons omit the required capacity and layer controls

Location: `section33/features/hidden.py:22`, `section33/probes/pipeline.py:329`, and `section33/cli.py:337`.

Hidden features use 64 projections of pre- and post-block states at eight layers: 1,024 columns. MLP summaries cover all layers. The primary B/B+M specifications have no internal dimension budget. The secondary comparison uses PCA-limited `H_512` against `H_all_plus_M`, which also changes the hidden representation by removing that PCA limit.

`H_M_256_256`, `H_768`, and `H_nonlinear` specifications exist, but the final evaluation loop does not evaluate them. It also does not execute the five-seed robustness path or compare normalized MLP input against matching hidden representations.

**Effect:** a gain could reflect additional layer coverage, readout capacity, or a weaker hidden bottleneck. It would not isolate the planned MLP-feature comparison.

**Required correction:** evaluate the matched-budget and additive comparisons, with the same hidden block on each side; match layers and positions; include the nonlinear controls and declared seed treatment. Save a feature/dimension manifest for each fitted model.

### F7 — High: insufficient cohorts are accepted without enforcing the design

Location: `section33/data/cohort.py:185`, `section33/data/cohort.py:213`, and `section33/cli.py:120`.

The sampler skips absent relations and takes fewer rows when a quota cannot be met. Split assignment fills the available subjects without requiring the requested sizes. The CLI rejects an empty cohort, but accepts an undersized cohort. It also does not enforce the planned complete PopQA exclusion list. Split assignment does not maintain the declared relation/stratum balance.

**Verified:** the manifest requests 183 test subjects but records 50. Mother is absent. The current source is the legacy PopQA pool, which the plan reserves for a separate check.

**Required correction:** distinguish pilot mode from confirmatory mode. Confirmatory mode must reject missing relations, insufficient quotas, incomplete exclusions, split overlap, and unmet power requirements. Enforce balanced allocation or explicit target-population weights. Keep this inspected cohort exploratory.

### F8 — High: secondary decision intervals are not multiplicity-adjusted

Location: `section33/cli.py:387` and `section33/cli.py:407`.

Holm correction is applied to four secondary AUROC p-values. The within-question comparison is outside that family. Its decision uses an ordinary 95% interval, although the decision text calls it adjusted. M-only equivalence also uses an ordinary interval. The planned primary-first hierarchy is not enforced; Track B can independently emit a generic primary verdict.

**Effect:** the code can claim a secondary gain or equivalence without the simultaneous uncertainty control required by the plan.

**Required correction:** freeze the full claim family and hierarchy, including the selected concentration test. Use appropriate corrected tests and simultaneous intervals for interval-based decisions. Report Track A and Track B verdicts separately.

## 3. Other required corrections and missing analyses

### F9 — Calibration uncertainty needs correction

Locations: `section33/probes/calibration.py:135` and `section33/analysis/evaluation.py:293`.

The bin residual mean is weighted by observations. Its variance correction instead uses an unweighted variance of cluster means. These are different estimators when subjects contribute different numbers of answers to a bin. In the bootstrap, repeated draws of the same subject retain the original cluster ID, so the correction collapses separate bootstrap copies into one cluster.

Use a subject-cluster, design-aware variance estimator for the actual weighted bin mean. Assign distinct IDs to bootstrap copies. Validate the resulting guardrail on controlled clustered data. The current guardrail should not support a formal calibration claim.

### F10 — The dominance mechanism is not an executed experiment

Locations: `section33/analysis/interventions.py:116` and `section33/cli.py`.

The intervention module provides tensor transformations, but no CLI path installs them in generation, measures changed answers, or runs the separate causal cohort. The final evaluation pools MLP families and does not test the selected concentration statistic in the predicted direction after controls.

The flatten/sharpen helper also transforms `abs(s)`, not contribution mass `abs(s_j) * norm(v_j)`. It is sensitive to function-preserving channel rescaling.

**Verified:** two equivalent parameterizations both produce output `[1, 10]`. Applying the same flatten intervention produces `[1, 10]` for one and approximately `[0.836, 26.428]` for the other. Thus this intervention does not currently test the invariant contribution-mass mechanism.

Correct the intervention definition and connect it to actual model outcomes with identity and matched random controls. Also correct removed-mass accounting for partial attenuation. Predictive utility can be reported separately; a claim that dominant interactions explain confidence needs these additional tests.

### F11 — Difficulty controls and final reporting are incomplete

Locations: `section33/analysis/difficulty.py`, `section33/probes/calibration.py`, and `section33/cli.py`.

Within-question AUROC is connected to final evaluation. Cross-fitted difficulty residuals, a strong nonlinear difficulty baseline, relation/difficulty subgroup comparisons, negative controls, reliability plots, and risk–coverage reporting are not connected to the final workflow. Helper functions alone do not supply their results. Separate literal-W2 and gated-contribution conclusions are also absent from the final decision file.

This leaves the central difficulty part of section 3.3 unresolved. The valid target is added predictive information after difficulty control. Unconditional independence from difficulty is neither established nor an appropriate requirement for calibrated probabilities.

### F12 — Freeze, provenance, labels, and resume checks are not enforced

Locations: `section33/cli.py:326`, `section33/cli.py:446`, `section33/config.py:137`, and `section33/models/collect.py:401`.

Evaluation refits using current code without verifying a frozen development selection, complete adjudication, or the stored protocol. The protocol hash excludes prompt text defined outside the protocol object. The audit checks for two forbidden strings and the presence of a hash, rather than checking all provenance and stage gates. The saved audit reports failure because a protocol hash was missing at that audit's time; the current protocol file has a hash, so this is not proof that it is still missing.

Fitted probes and calibrators are not persisted by the evaluation command. Adjudication helpers exist, but final labels are not required to have two independent reviews. Track A assigns every question in a batch the hash of `prompts[0]`. Resume marks a question complete from its label file alone, although labels, confidence, and features are appended separately. An interruption can therefore leave a permanently incomplete question.

Require a frozen manifest containing actual model/tokenizer revisions, prompt/template hashes, feature definitions, selected pipelines, label policy, and data hashes. Verify it before evaluation. Save the fitted objects and complete prediction provenance. Resume only complete, consistent records. These checks must fail on missing or mismatched artifacts.

## 4. Match to section 3.3 and the plan

| Required question or component | Current status | Can it support the intended conclusion? |
| --- | --- | --- |
| Small probe over MLP interaction features | Implemented; gated math and several invariance tests pass | Useful foundation |
| Literal W2 versus actual gated computation | Separate feature families exist | Separate final claims still needed |
| MLP adds value beyond D+P+H | P is empty; required controls omitted | No |
| Before-answer prediction | Prompt activations are correct; answer length enters D | No |
| MLP replaces hidden states | Comparison exists; capacity and interval problems remain | No |
| Information beyond difficulty | Within-question analysis exists; other controls are incomplete | Not established |
| Reliable confidence probabilities | Held-out calibration exists; weighting and variance problems remain | Not established |
| Dominant contributions explain confidence | Helpers exist; intervention definition and execution incomplete | No |
| Fresh, powered test | Legacy pilot; 50 test subjects; power path broken | No |
| Replication across two models | Configured; only Qwen decision found | No |
| Frozen, reproducible final decision | Partial manifests; gates not enforced | No |

Useful implemented parts include the gated MLP reconstruction, separate reference-free feature families, exact candidate replay, subject-grouped inner folds, fold-internal preprocessing, held-out calibration, and paired subject bootstrap scaffolding. These should be retained while the experiment connections are repaired.

## 5. Verification performed

The following existing checks passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  -q -p no:cacheprovider \
  tests/test_section33_contracts.py tests/test_signal_contract.py
```

Result: **29 passed in 77 seconds**. These tests do not establish protocol compliance. For example, the end-to-end fixture contains a probability column, but never asserts that P consumes it or that P is nonconstant.

Additional CPU checks reproduced F1, F2, F3, F5, and the intervention invariance failure. Saved artifacts confirmed the sample loss and constant P score. Evidence is in `report/SECTION33_REVIEW_EVIDENCE.json`; the reproduction script is `report/section33_review_checks.py`.

## 6. Order of work before a new final test

1. Repair feature selection, Track A metadata, stable candidate joins, and multiplicity. Add integration tests with realistic collector output, duplicate decoded strings, and interrupted writes.
2. Repair power estimation, calibration uncertainty, corrected secondary decisions, and strict cohort validation. Test known-null, useful-effect, harmful-effect, and insufficient-data cases.
3. Complete the capacity/layer controls, difficulty analyses, family ablations, and persisted pipeline workflow. Complete the causal runner if making the dominance claim.
4. Use the current inspected run for development and regression checks. Construct the fresh cohort, complete blind label review, freeze all choices, and reserve an untouched final set. Determine its size from the repaired development-only power analysis.
5. Run the primary Track A experiment and required Track B checks, then the replication model. Publish the declared verdict for each claim, including inconclusive results when warranted.

**Answer to the user’s question:** the implementation follows parts of the plan, but it does not yet implement the experiment faithfully enough to answer section 3.3. The present evidence supports neither “MLP features give a useful improvement” nor “a useful improvement is ruled out.” A corrected, sufficiently powered experiment can distinguish those outcomes within the declared model, task, and probe scope; it cannot guarantee a conclusive outcome in advance.
