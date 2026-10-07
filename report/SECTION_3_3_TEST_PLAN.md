# A decisive test of MLP features for confidence

**Scope:** `meet.md`, section 3.3, and the corresponding proposal in `meet.original.txt`.
**Status:** proposed experiment; no new model runs or confirmatory results.
**Prepared:** 7 October 2026.

## 1. The decision this experiment must resolve

The proposal contains several claims. A single AUROC comparison cannot resolve all of them.

1. MLP interaction features contain useful information about answer correctness.
2. That information improves a confidence estimator beyond strong hidden-state and probability baselines.
3. The improvement remains after question difficulty is controlled.
4. Concentrated MLP interactions indicate reliable recall; competing interactions indicate uncertainty.
5. The resulting probabilities are useful confidence estimates, rather than only useful rankings.

**Recommended primary question:** On unseen subjects, does adding reference-free MLP features improve a strong confidence estimator by a practically useful amount?

**Recommended primary setting:** Predict whether the model's next greedy answer will be correct, before it produces that answer. This gives a direct continuation of the pilot without reference-answer inputs.

**Required second setting:** Estimate correctness after the model produces a candidate answer. Several sampled answers to the same question permit a stronger control for question difficulty. Conclusions from this setting must remain answer-specific; they cannot be presented as pre-answer results.

A definitive answer must have a stated scope. This plan can establish a useful gain, rule out a gain of a specified size, or identify insufficient evidence for the tested models and data. No finite test can establish independence from every possible difficulty factor or reject every possible MLP architecture.

### Correct the meaning of difficulty independence

The target should be **information beyond difficulty**, not a confidence score uncorrelated with difficulty. If difficult questions have lower success rates, a calibrated estimator should usually give them lower confidence.

For a score calibrated within difficulty groups, `E[p | D] = E[Y | D]`. If success rates vary with difficulty `D`, the mean calibrated score must vary too. A requirement that `p` be independent of `D` would conflict with this condition. This is a mathematical design constraint, not an experimental finding.

Therefore, do not require an adversary to remove every difficulty signal. Test conditional predictive value, robustness across difficulty groups, and discrimination between correct and incorrect answers to the **same question**.

## 2. What the current pilot does and does not establish

The saved main-target scores are 0.871 for hidden states and 0.907 for hidden states plus MLP summaries. The observed difference is +0.037, with a paired interval of [+0.023, +0.051]. MLP summaries alone reach 0.863; their difference from hidden states is inconclusive.

These are exploratory results. The strongest features use reference-answer information. The selected layers use the same evaluation data. There is no untouched final test set. The confidence target is a threshold on repeated sampled success, not one answer's correctness. Calibration and the proposed dominant-interaction mechanism remain untested.

Use the historical data for implementation checks, feature development, and conservative variance estimates. Do not reuse its reported improvement as an unbiased power estimate. Do not relabel a new split of those questions as a final test.

Sources: `report/AUDIT.md`, `report/REPORT.md`, `results/pilot7b_frac_extra.csv`, and `results/audit_checks.json`.

## 3. Define the actual MLP computation before selecting features

### 3.1 Separate the meeting's description from the implemented mechanism

For the installed Qwen2 implementation, let `x` be the normalized input to the MLP:

```text
a = W_gate x
b = W_up x
s = SiLU(a) * b           # elementwise product
o = W_down s
v_j = column j of W_down
c_j = s_j v_j             # contribution of channel j to the MLP output
o = sum_j c_j
```

The installed module uses no bias in these three projections. Verify this contract separately for each replication model. Save both the module input and its corresponding residual-stream input; normalization changes the comparison.

**Important correction:** `x dot v_j` is a computable similarity, but it is not the coefficient used to select channel `j` in this gated MLP. The actual coefficient is `s_j`. The meeting's literal query-to-W2 proposal and the actual gated computation must have separate feature names and separate results.

The key-value interpretation and ROME motivate this experiment; neither establishes that a dominant W2 similarity is calibrated confidence in Qwen. See [Geva et al., key-value memories](https://aclanthology.org/2021.emnlp-main.446/) and [Meng et al., ROME](https://rome.baulab.info/).

### 3.2 Freeze four MLP feature families

| Family | Definition | Purpose |
| --- | --- | --- |
| Literal W2 similarities | Raw `x dot v_j`, plus cosine similarities with explicit norm controls | Test the wording of section 3.3 directly |
| Gate and selection features | `a`, `SiLU(a)`, `s`, and their distribution summaries | Test the actual routing computation |
| Contribution concentration | Distribution of `e_j = abs(s_j) * norm(v_j)` | Test whether a few output contributions dominate |
| Output and direction features | `norm(o)`, `norm(o)/norm(x)`, `cos(o,x)`, and cancellation measures | Separate concentration from size and direction |

For each nonnegative mass vector, set `p_j = e_j / sum(e)`. If total mass is zero, set a missing/zero-mass indicator and use a documented fixed value for each statistic.

Freeze these concentration summaries: normalized entropy `-sum(p log p)/log(m)`, top-1 mass, top-5 mass, top-1 minus top-2 mass, and effective channel count `1/sum(p^2)`. Here `m` is the number of channels. For signed dot products, summarize positive and negative parts separately, and retain a signed total. Do not silently convert signed evidence into a probability.

Use contribution magnitude as the main concentration definition. Raw selection magnitudes alone depend on channel scaling. Under the transformation `W_up[j,:] *= k` and `W_down[:,j] /= k`, for positive `k`, the function and each `c_j` stay unchanged. Raw `s_j` concentration can change. Verify this invariance numerically before using concentration as a mechanism explanation.

Keep the pilot's 12 reference-free summaries as a replication family. Remove all five reference-dependent summaries from every confirmatory feature set. Preserve their old names only in the historical results.

### 3.3 Feature positions and information boundaries

- **Before-answer track:** last prompt token only. No generated token, reference token, answer alias, object popularity, or correctness label can enter the feature extractor.
- **After-answer track:** last content token of the model's own candidate answer, before a confidence request. The reference answer is still unavailable. Hidden and MLP features use the same position.
- **Secondary positions:** last subject token and mean over generated content tokens. Select these only in development; do not choose a position after viewing test results.
- Record pre-block hidden state, normalized MLP input, and post-block hidden state. Compare each with corresponding MLP features at the same layer and position.
- Candidate-token alignment is a separate secondary family. It may use the model's own tokens or its own top-k next-token candidates, never the reference token. Raw projections are diagnostic values, not exact final-logit contributions.

MLP features are deterministic functions of internal states and fixed weights. Thus, the claim concerns accessibility to a specified probe class and data budget. It cannot mean that MLP features add information unavailable to an unrestricted function of those same internal states.

## 4. Data that can support a new answer

### 4.1 Inventory checked in this workspace

The local cached PopQA snapshot contains 14,267 questions and 12,244 subject IDs. The original four-relation subset contains 1,873 questions and 1,799 subjects. The union of subjects in saved label files contains 1,066 IDs.

After excluding those subjects, the original domain has only **733 questions from 733 unseen subjects**:

| Relation | All available questions | Questions on unseen subjects |
| --- | ---: | ---: |
| Father | 570 | 237 |
| Mother | 187 | 0 |
| Place of birth | 584 | 276 |
| Occupation | 532 | 220 |

These counts came from the cached Arrow data and all `results/*labels.csv` files. The snapshot is saved in `report/SECTION_3_3_DATA_INVENTORY.json`. Before freezing the split, inspect other logs and exported data for additional exposure. The zero remaining mother questions means a fresh four-relation test cannot come from this pool alone.

Reserve the 733 questions as a separate legacy-domain check. They are not a substitute for a powered test and cannot support a fresh mother-relation result. Do not fill the gap with different relations while retaining the original population claim.

### 4.2 Construct a fresh person-question cohort

Start with **6,000 new subjects**, one primary question per subject. Exclude every subject in the cached PopQA snapshot and all previously used subjects. This conservative rule also keeps the 733-question check separate.

Use equal relation quotas and equal popularity-quintile quotas within relation. The target is the resulting balanced research population, not the distribution of all real user questions. Fix the popularity measurement window before collection. Give missing page-view metadata an explicit stratum.

Use a dated Wikidata snapshot and documented statement sources for [father, P22](https://www.wikidata.org/wiki/Property:P22), [mother, P25](https://www.wikidata.org/wiki/Property:P25), [place of birth, P19](https://www.wikidata.org/wiki/Property:P19), and [occupation, P106](https://www.wikidata.org/wiki/Property:P106). Verify that the new pool can meet the quotas before committing to the sample size.

Exclude ambiguous subject names without a fixed disambiguation rule, unresolved conflicting facts, and questions whose wording has no stable answer. Accept documented multiple occupations; freeze the semantic granularity policy. Exclude mythical or disputed subjects from the main cohort and identify this scope difference from the pilot. Do not remove questions because a model gets them wrong.

A new data snapshot does not imply a fact is new to model pretraining. Record provenance and relevant dates. Report likely recent or changed facts separately. The study tests held-out evaluation subjects, not absence from pretraining.

### 4.3 Assign disjoint splits before model collection

| Split | Initial subjects | Permitted use |
| --- | ---: | --- |
| Training | 2,000 | Fit probes and difficulty models |
| Development | 800 | Select features, layers, hyperparameters, power assumptions, and frozen decision rules |
| Calibration | 800 | Fit probability calibration and choose abstention thresholds after probe selection |
| Final test | 2,400 | One final evaluation of frozen methods |

Increase the final-test allocation, if the power analysis requires it, to a cap of 6,000 subjects. This makes the maximum fresh cohort 9,600 subjects. The allocation is an initial design, not a claim that 2,400 subjects are sufficient.

All answers, paraphrases, relations, and model outputs for a subject must stay in its split. Group aliases and near-duplicate facts. Record subject–answer entity overlap; run an answer-entity-disjoint sensitivity check where feasible. Shared occupation labels alone must not connect the entire data set into one group.

A split manifest contains subject IDs, question IDs, source revisions, relation, popularity stratum, and a hash. Seal test labels from the analysis code until the models, calibration rules, and report specification are frozen. A filesystem hash records the freeze; it is not a substitute for disciplined access control.

## 5. Models, answers, and trustworthy labels

### 5.1 Model scope

Use [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for the primary test. Use [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) as a required replication on another model family. Pin weight revisions, tokenizers, chat templates, software versions, numerical precision, and generation settings.

Use each model's documented chat template and the same semantic task. Adapt hook contracts per architecture. Run one model at a time on the existing machine. Quantization changes the tested model; do not introduce it only for one method. If replication cannot run, limit the conclusion to Qwen rather than selecting an easier replacement after results are known.

Freeze both models' protocols before opening either final test. Allow model-specific fitting on their own training/development outputs under the same search rules. Map the fixed layer set by relative depth. Keep demonstration subjects out of every evaluation split. Freeze any model-specific prompt adaptations before test collection.

### 5.2 Two distinct outcomes

**Track A, before-answer:** one greedy answer per question. The label is whether that entire answer is correct. Use the last prompt position to predict this label. Fix the short-answer prompt, examples, stop condition, and a 32-token content limit. Preserve refusals, empty answers, and truncations as recorded outcomes.

**Track B, after-answer:** eight independent sampled answers per question, with temperature 1.0 and top-p 1.0. The label is correctness of each candidate. Extract features by replaying the exact prompt plus the model's generated tokens. No new answer may replace the original during feature collection.

Weight questions equally, then weight their eight samples equally. Cache duplicate candidate strings or token sequences for efficiency, but retain their sampling multiplicities in the estimand. Eight answers from one subject are not eight independent subjects.

The mean of eight labels is a noisy estimate of sampled success. Do not threshold it at 0.5 and present it as precise knowledge. If a fraction-correct analysis is wanted, collect a separate fixed 30-sample replication subset and use binomial counts directly. It is secondary and cannot rescue a failed primary test.

### 5.3 Confidence requests

For Track A, ask for the probability that the model's next greedy short answer to the same question will be correct. Use identical factual context and examples in the answer and confidence branches. Extract probe features from the answer branch, before any confidence instruction.

For Track B, show the exact candidate answer and ask for its probability of being correct. Use the same factual context. Collect confidence in a separate branch after recording answer features.

Save raw confidence text. Use a strict parser accepting only the requested numerical format in [0,100]. Mark failures as missing. A train-fitted median plus a missingness indicator is the operational baseline; also report valid-only results on a common subset. Never encode a parse failure as genuine zero confidence.

### 5.4 Label policy

Use normalized exact entity/alias matches and token boundaries. Do not accept an alias merely because it appears inside an unrelated word. Define rules for multiple answers, contradictory additions, birth-place granularity, and occupation synonyms before annotation.

For the final test, two reviewers independently check each unique question–candidate answer and resolve disagreements without seeing probe scores. Review reference facts as well as model text. Review calibration labels with the same policy. Development may use automatic matches with manual review of unresolved cases, followed by a stratified audit of accepted and rejected matches.

Report agreement, adjudication rate, matcher precision/recall against adjudicated labels, and label changes. A model-based judge can triage cases but cannot be the sole final authority. Label correction must apply across all methods, including inconvenient failures.

## 6. Baselines and comparisons that isolate MLP value

### 6.1 Required feature sets

| Code | Inputs | What it controls |
| --- | --- | --- |
| D | Question-only metadata: relation, subject popularity, prompt/name length, language count, collision count, missingness | Observable difficulty |
| V | Stated confidence, raw and recalibrated | Prompted self-assessment |
| P | Reference-free token probability features | Output uncertainty |
| H | Hidden states at matched positions; layer input and output variants | Standard internal probe |
| M | Each frozen MLP family, alone | Replacement claim |
| H+M | The same hidden features plus MLP features | Direct extension of the pilot |
| B | D+P+H, selected with development data | Strong practical comparator |
| B+M | Identical B inputs plus MLP features | Main confirmatory comparison |

Before-answer P uses top-1 probability, top-1/top-2 margin, entropy, and fixed top-k mass. After-answer P additionally uses sequence log probability, mean/minimum token log probability, token entropy summaries, and answer length. No feature uses the reference probability, reference rank, object popularity, or reference alias count.

Allow D a nonlinear baseline, such as gradient-boosted trees, so a weak surface model does not create an artificial residual signal. Fix its tuning budget alongside the probe budget. Inspect the actual implemented inputs; proposed metadata absent from the pilot must be collected before use.

### 6.2 Fair capacity and layer controls

Use standardized L2 logistic regression as the main readout. Give H and H+M, and B and B+M, the same development search budget. Add one small nonlinear readout as a required robustness control, using the same parameter budget and training schedule on both sides.

Freeze a 512-dimensional internal-feature budget for the matched-capacity comparison:

- H control: 512 dimensions from hidden states across the same layers.
- H+M: 256 hidden dimensions plus 256 MLP dimensions.
- Use training-fitted PCA for high-dimensional vectors. Standardize blocks on training data.
- Also report the additive comparison retaining all 512 hidden dimensions plus 256 MLP dimensions.
- Compare that additive model with a 768-dimensional hidden-only control and with a hidden-plus-nonlinear-hidden-summary control.
- Keep D and P identical outside this internal-feature budget.

A fixed dimension budget can discard useful hidden information. Therefore, report both matched-budget and additive comparisons. Do not call a PCA bottleneck failure proof that raw MLP features have no signal.

For layer selection, start with all-layer summaries and a fixed evenly spaced layer set. Select a single-layer variant on development only. A combined multi-layer MLP model must face a multi-layer hidden baseline, not only the pilot's selected single hidden layer.

Fit scalers, PCA, imputation, feature selection, and classifiers inside every training fold. No inner-validation record may fit its own preprocessing. Use subject-grouped folds. Save the exact fitted pipeline and feature order.

Use three grouped inner folds and select configurations by mean Brier score, with AUROC as a fixed tie-breaker. The initial logistic regularization grid is C = {0.001, 0.01, 0.1, 1, 10}. Allow at most 24 prespecified configurations for B and 24 for B+M in each track; divide the augmented budget across feature families before selection. Count feature/layer alternatives in that budget. The robustness neural readout has one hidden layer, learning rates {0.0001, 0.001}, weight decay {0.0001, 0.001}, at most 100 epochs, and patience 10 on inner-validation Brier score. Match its trainable parameter budget across comparisons. Freeze the configuration list after the smoke test, before final evaluation.

### 6.3 Calibration and budget fairness

After method selection, fit a calibration map on the calibration split only. Compare identity, logistic/Platt scaling, and isotonic calibration through internal grouped cross-validation on that split. Refit the chosen map on the full calibration split. Apply the same selection process to each baseline and augmented model. Report before- and after-calibration scores.

Calibration is the relation between forecast probabilities and success frequencies; simple held-out recalibration is a relevant baseline. See [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html). This source motivates calibration checks; it does not establish that any chosen method will work here.

Use fixed probe seeds 0–4 for stochastic models. The primary prediction is the mean of the five fitted predictions, with identical treatment of controls; report each seed as sensitivity evidence. Logistic fits may be deterministic. Never select the best seed on test data.

## 7. Difficulty control that can distinguish the hypotheses

### 7.1 Conditional prediction beyond measured difficulty

Compare D with D+M, and B with B+M, on unseen subjects. Include relation and flexible popularity effects in D. Report performance by the training-defined popularity groups, relation, and train-fitted difficulty-score groups.

Train-defined bin boundaries stay fixed on test data. Report counts and class balance. Do not draw conclusions from an AUROC for a one-class group. Report the overall balanced-population result and prespecified per-group results; do not choose a favorable subgroup afterward.

### 7.2 Compare different answers to the same question

For each Track B question containing both correct and incorrect sampled answers, compute the probability that a correct answer receives a higher score. Count ties as one half. Average this within-question AUROC equally across eligible questions.

Compare B+M against B and V using this measure. Report the number and fraction of eligible questions. This estimate concerns mixed-outcome questions only. Also report Brier score on **all** sampled answers so selection of mixed questions cannot hide poor confidence on always-wrong questions.

Question-only difficulty is exactly shared within each question. Candidate length, probability, and answer identity can still differ, so P and answer-length controls remain necessary. A positive result shows answer-specific information beyond fixed question properties. It does not establish pre-answer difficulty independence.

### 7.3 Learn under difficulty control

Use relation-by-popularity-balanced batches or weights, fixed from the sampling design. This controls representation in training but does not erase difficulty information.

For a second Track B probe, use a within-question pairwise loss on correct/incorrect answer pairs. Sample questions uniformly, then pairs within a question, to prevent prolific subjects from dominating. Apply the same loss and pair set to H/B controls. Calibrate the resulting score afterward on the separate calibration split. Compare this probe with the ordinary correctness-loss probe.

Do not use an unconditional difficulty adversary as the main test. Such an objective can remove information needed for accurate confidence. It may be explored later, with a separate claim and utility check.

### 7.4 Residual and robustness checks

For binary correctness, fit a cross-fitted difficulty predictor of binary correctness. For sampled fraction correct, fit a binomial model of the sample counts. Match the target on both sides before computing any residual.

Residual correlations are secondary diagnostics. A misspecified surface model can leave difficulty in its residual. They cannot establish causal independence.

Reserve a fixed subset for two manually reviewed question paraphrases. Keep all paraphrases in the subject's split. Evaluate unchanged probes and calibrators on these prompts. Also test a low-popularity transfer condition using only training/development choices. Report shifts in both accuracy and calibration.

## 8. A direct test of the dominant-interaction mechanism

This part is required before saying that section 3.3's proposed mechanism is supported. Predictive improvement alone is not sufficient.

### 8.1 Observational test

Predefine the layer pool, positions, and concentration statistics in Section 3. Fit a concentration-only probe and a probe of norms/directions without concentration. Compare B+concentration with B+norms/directions and B+all MLP features.

The directional prediction is explicit: higher top-1 mass and lower normalized entropy should accompany higher correctness after the stated controls. Test this within Track B questions and across held-out questions with difficulty controls. Analyze confident wrong answers separately.

Select one primary concentration statistic and one layer-aggregation rule on development. Treat other statistics as corrected secondary tests. If only output direction helps, the practical feature claim can succeed while the dominance explanation fails.

### 8.2 Numerical and identity controls

- Hook identity: without an intervention, reconstructed `W_down s` must match the module output.
- Positive channel rescaling: predictions and contribution-based concentration must remain unchanged within numerical tolerance.
- Permutation control: shuffle channel identities jointly with the corresponding weight columns. Function and identity-free summaries must remain unchanged.
- Wrong-question control: replace MLP features with features from another training subject in the same difficulty stratum. Refit the same probe. This is a shortcut diagnostic, not an exact conditional-independence test.
- Reference-swap control: changing the hidden evaluation-only reference file must not change collected features or generated answers.

These checks can reject an implementation or a fragile statistic before expensive causal tests.

### 8.3 Controlled interventions

Use an additional, preregistered causal cohort disjoint from training and the predictive final test. Start with 400 subjects and increase, before viewing causal results, according to a separate paired power calculation. Use both initially correct and initially incorrect examples; retain the full cohort in the main analysis.

Select layers using development only. Apply interventions at the last prompt position and regenerate the answer:

1. Identity intervention: rewrite the original activation unchanged.
2. Remove or attenuate the highest-contribution channels, with fixed top-k and attenuation settings.
3. Remove random channels with approximately matched removed contribution mass and output perturbation norm.
4. Flatten or sharpen the contribution mass distribution, preserving signs and channel directions. Use exponents 0.5, 1, and 2.
5. Repeat flattening/sharpening with total MLP-output norm restored to its original value, when the norm is nonzero.

Predeclare the matching caliper, for example 10%, on development. If matched random controls cannot be found, report that comparison as unidentifiable under this design. Do not accept larger perturbations for the top-channel intervention without disclosure.

Measure changes in answer correctness, generated-answer identity, concentration, model probabilities, and the frozen probe's confidence. Reference-answer probabilities may be used **only as evaluation outcomes** here; they cannot choose an intervention or enter a probe.

Interpretation must be narrow. Greater damage from top-channel removal supports causal participation of those contributions. It does not prove one channel stores one fact. Flattening and sharpening also change output direction; norm matching does not isolate concentration perfectly. If sharpening increases confidence without increasing success, the simple dominance-to-confidence explanation is weakened.

The mechanistic conclusion requires agreement between held-out concentration tests, invariance controls, and controlled interventions. Intervention effects alone can show that computation matters without showing that its concentration estimates confidence.

## 9. Metrics and a frozen statistical analysis

### 9.1 Primary estimands

The primary comparison is calibrated B+M against calibrated B in Track A on the fresh final test. Define all improvements as positive numbers:

```text
Delta_AUC   = AUROC(B+M) - AUROC(B)
Delta_Brier = mean((B - Y)^2) - mean((B+M - Y)^2)
```

Brier score is a proper probability score. It reflects both discrimination and calibration, so an improved Brier score alone must not be called proof of improved calibration.

Report H+M minus H as the direct pilot replication, M minus H as the replacement test, and comparisons with V and P. The main claim must survive B, not only the weakest baseline.

### 9.2 Required secondary measures

- Log loss, with a fixed numerical clipping rule used by every method.
- AUROC and precision–recall AUC, with prevalence reported.
- Reliability plots, calibration intercept/slope, and fixed-bin calibration error.
- Within-question AUROC and all-answer Brier score for Track B.
- Risk–coverage curves: error rate among retained answers at 25%, 50%, and 75% coverage.
- Coverage and error at an abstention threshold fixed on calibration data for a target 10% error rate. Report the actual test error and its interval; do not claim a guarantee.
- Added inference latency, memory, feature-storage size, and any extra model passes.

Choose common probability-bin edges before test evaluation. Start with ten equal-width bins; use calibration-only counts to freeze any adjacent-bin merges. Show test counts. Do not change bins to make a reliability plot look better.

For a formal calibration-improvement claim, compare a prespecified, variance-corrected squared bin-calibration error in addition to Brier score. For residuals `r = Y - p` in bin b, use the weighted sum of `mean(r)^2 - sample_variance(r)/n_b`. Do not clip negative estimates before comparing methods. Sparse bins and the chosen resolution limit interpretation. Report ordinary reliability plots alongside this statistic.

The displayed variance correction assumes independent, equally weighted observations. For repeated answers or design weights, replace `sample_variance(r)/n_b` with a subject-cluster, design-aware estimate of the variance of the bin mean. Freeze this estimator and its bootstrap procedure in development. Do not treat sampled answers as independent when estimating calibration error.

### 9.3 Uncertainty and multiplicity

Use 10,000 paired bootstrap samples of **subjects**. Keep all answers and paraphrases for a subject together and use identical draws for both methods. For the balanced target population, resample subjects within the frozen relation/popularity strata, retaining the target stratum weights. Report the bootstrap seed.

These intervals describe test-population sampling with fitted models held fixed. Report training-seed sensitivity separately. A smaller prespecified outer resampling exercise can refit whole pipelines to assess training variability; do not describe the ordinary test bootstrap as including it.

Use hierarchical testing. First test the primary Track A B+M versus B comparison. Its joint practical-success rule requires both endpoints, so it is an intersection requirement, not selection of whichever metric wins. Use Holm correction at familywise 0.05 for the separately claimed secondary comparisons, including Track B gains, M-only superiority, and the selected concentration test. Publish adjusted tests and conservative simultaneous intervals when using interval-based secondary decisions.

Correct intervention-family comparisons separately. Subgroup plots and unselected feature variants are exploratory unless they have a separately registered family. A secondary result cannot retroactively replace a failed primary result.

## 10. Sample size and explicit yes/no/inconclusive rules

### 10.1 Practical effect sizes

The following are **proposed study decisions**, not literature constants:

| Quantity | Smallest gain worth claiming |
| --- | ---: |
| Main AUROC gain | +0.020 |
| Main Brier-score reduction | +0.005 absolute |
| Track B within-question AUROC gain | +0.030 |
| Allowed increase in fixed-bin calibration error | +0.010 absolute |

Freeze these margins before the final-test collection or evaluation. If operating costs justify different margins, change them while only development data are visible and document why.

### 10.2 Power analysis before locking the final test

Use new development predictions without reference-answer features. Simulate paired outcomes and predictions at the **subject** level, preserving relation/popularity strata and within-question answer dependence. Use several plausible prevalences and training-variance scenarios. The old gold-dependent gain is not the assumed new effect.

Evaluate final-test sizes of 800, 1,600, 2,400, 4,000, and 6,000 subjects. Choose the smallest feasible size that provides at least 90% simulated probability of the intended decision under both:

- A useful alternative, such as true AUROC gain +0.040 and Brier reduction +0.010.
- A negligible alternative, such as zero gain, for ruling out the prespecified useful margins.

Also simulate the probability of an inconclusive result near the margins. A test cannot have 90% probability that its lower bound exceeds a margin when the true effect equals that margin.

Check class counts, not just total N. Use at least 300 correct and 300 incorrect **subjects** in the main test as an initial planning floor, and at least 300 mixed-outcome subjects for a within-question analysis. These floors do not replace the power calculation. Plan counts from development prevalence; do not collect until the desired test outcomes appear.

If 6,000 subjects are insufficient, the study must expand before unblinding, accept a wider inconclusive region, or explicitly reduce the scope. Repeated answers cannot substitute for missing independent subjects in Track A. Do not stop when a p-value first becomes favorable.

### 10.3 Primary decision rules

Let L and U denote the paired 95% interval bounds.

**Useful predictive improvement supported:** `L(Delta_AUC) > 0.020` and `L(Delta_Brier) > 0.005`. Require all information-boundary and label checks to pass. Require the augmented score's calibration-error increase to have an upper bound below 0.010 before calling it a practical confidence improvement.

**Useful improvement ruled out for this protocol:** `U(Delta_AUC) < 0.020` and `U(Delta_Brier) < 0.005`. This rules out the registered practical effect, not every tiny improvement and not every possible MLP probe.

**Harm supported:** an interval for a primary improvement lies wholly below zero. Report the affected endpoint explicitly.

**Mixed or inconclusive:** every remaining pattern, including better ranking without better probability scoring, intervals crossing the practical margins, or inadequate calibration evidence. State exactly which part remains unresolved.

To claim **improved calibration itself**, also require a corrected positive improvement in the prespecified calibration-error statistic. To claim **reliable absolute probabilities**, preregister an operational tolerance and demonstrate it at the tested bin resolution; a relative improvement alone is insufficient.

For the **within-question claim**, require the multiplicity-adjusted lower bound on B+M minus B within-question AUROC to exceed +0.030. Also require an upper bound below 0.002 on any increase in all-answer Brier score. An adjusted upper bound below +0.030 rules out the registered within-question gain. Otherwise, this component is inconclusive. Include these requirements in its own power simulation; a plausible useful alternative is +0.060 within-question AUROC.

For **MLP-only replacement**, report superiority separately from equivalence. An interval crossing zero does not establish equivalence. A corrected interval entirely inside [-0.020, +0.020] establishes practical AUROC equivalence at that registered margin, with probability-score checks still required.

For the **dominance mechanism**, the selected concentration statistic must have the predicted held-out direction after controls, with an adjusted interval excluding zero. Top-contribution attenuation must impair success more than matched random attenuation, with a corrected interval excluding zero. Invariance tests must pass, and flatten/sharpen results must not systematically increase confidence while reducing success. Meeting these conditions supports the bounded mechanism tested here; inconsistent or imprecise results leave it unconfirmed. Report each component rather than hiding conflicts in an aggregate score.

For the main practical claim, repeat the frozen procedure on the second model. Report model-specific decisions. A cross-model conclusion requires the primary criteria in both models. Do not pool a large gain in one model with a failure in the other and call the result universal.

### 10.4 Claim-to-evidence decision map

| Claim | Required evidence | Wording if evidence is absent |
| --- | --- | --- |
| Adding MLP improves confidence | Primary B+M gain, probability-score gain, calibration guardrail, valid reference-free inputs | No established practical confidence gain |
| MLP should replace hidden states | Corrected superiority of M over matched H | Replacement is unsupported or inconclusive |
| MLP adds information beyond difficulty | Conditional gains plus Track B within-question gain over B | Difficulty explanation remains viable |
| Dominant interactions indicate reliable recall | Directional concentration result, controls, and consistent interventions | Mechanism unconfirmed, even if prediction improves |
| Benefit generalizes | Frozen-protocol replication on the second model and stated shift tests | Restrict the conclusion to tested conditions |

A strong section 3.3 conclusion requires the practical, difficulty-control, and mechanistic evidence together. Otherwise, report the narrower result rather than a single unqualified yes.

## 11. Required ablations and failure checks

Freeze this list before the final test:

1. Literal W2 similarities versus actual selection/contribution features.
2. Concentration only, norm/direction only, and both.
3. MLP-only, hidden-only, and their combination.
4. Single-layer versus matched all-layer inputs.
5. Matched dimension budget and matched nonlinear readout.
6. B+M versus B with identical surface/probability features.
7. Ordinary versus within-question pairwise training in Track B.
8. Raw versus identically recalibrated scores.
9. Reference-free features versus the historical oracle features, with the latter reported only as a development diagnostic.
10. Label permutation, wrong-question feature replacement, reference-file swap, and hook reconstruction.

Failure cases must include concentrated but wrong predictions, diffuse but correct predictions, low-popularity subjects, ambiguous aliases, repeated identical wrong answers, and prompt paraphrases. Report their frequencies, not only selected examples.

## 12. Concrete implementation work in this repository

This is a plan, not a statement that the following commands or modules already exist.

| Existing component | Required change | Acceptance check |
| --- | --- | --- |
| `scripts/prepare_data.py` | Add fresh-cohort construction and immutable subject manifests under a new run directory | Zero subject overlap; quotas and provenance report |
| `scripts/collect.py` | Separate label/reference storage from feature input; support pre-answer and generated-answer positions | Reference-swap invariance; saved exact token IDs |
| `scripts/collect.py` | Add literal W2, gated, and contribution feature contracts | Reconstruct actual MLP outputs and pass scaling controls |
| `scripts/probe_eval.py` | Fit complete preprocessing pipelines inside every fold | Deliberate leakage tests fail; held-out records never fit transforms |
| `scripts/extra_probes.py` | Add fixed B/B+M, capacity controls, and within-question loss | Same subjects, positions, tuning budget, and weights |
| New calibration module | Separate calibration fitting and threshold selection | No test-label access; serialized calibration maps |
| New difficulty module | Train-only strata and question-level comparisons | Correct target residuals; mixed-question counts |
| New intervention module | Identity, attenuation, matched random, flatten/sharpen | Identity reproduces outputs; matched perturbation audit |
| `scripts/audit_artifacts.py` | Validate new schemas without altering historical files | Reproducible labels, joins, predictions, and metric differences |
| `tests/test_signal_contract.py` | Extend tests to actual positions, reference exclusion, replay, and invariance | CPU/unit tests plus a small model smoke test |

Suggested new run layout:

```text
runs/section33_v1/
  protocol/          frozen design, margins, search space, power report
  manifests/         subjects, splits, prompts, source revisions, hashes
  labels/            reference data, adjudications, generation outcomes
  features/          reference-free scalars, compressed vectors, schema
  probes/            preprocessing, fitted readouts, calibrators
  predictions/       one row per model/subject/question/answer/method
  evaluation/        paired intervals, subgroup counts, decision.json
  interventions/     settings, matched controls, regenerated outcomes
  report/            claim decisions, tables, plots, failure examples
```

At minimum, each prediction row needs model revision, subject/question/answer IDs, split, target, generation seed, feature position, probe ID, raw score, calibrated score, and label-version hash. Keep references in a separate interface from feature collection. Do not overwrite the pilot artifacts.

## 13. Execution stages and gates

| Stage | Work | Exit condition |
| --- | --- | --- |
| 0. Freeze definitions | Claims, target populations, feature equations, comparison list | Written protocol and traceable implementation specification |
| 1. Prove correctness | 50-question smoke test, replay, labels, hooks, leakage controls | Every data/feature contract passes |
| 2. Build development cohort | Fresh subjects, development collection, label audit | Feasible quotas; usable class balance; measured costs |
| 3. Develop and size | Select probes on development; simulate power and limits | Frozen methods, margins, sample size, and calibration policy |
| 4. Fit and calibrate | Final training; separate calibration and threshold selection | Serialized frozen pipelines and report specification |
| 5. Open predictive test | One evaluation on fresh subjects and the 733-question check | Complete intervals and registered decisions, including failures |
| 6. Replicate and test mechanism | Second model, difficulty tests, causal cohort | Separate mechanism and replication decisions |
| 7. Write verdict | Complete claim table, costs, and failure analysis | Bounded yes/no/inconclusive answer for every subclaim |

Development futility is a resource decision. If no reference-free MLP variant survives strong controls on development, stop with a development-level negative result. This does not count as a definitive held-out rejection. A formal negative answer still needs the powered frozen test.

If the primary final test is inconclusive, do not tune on it. Any follow-up requires a new versioned protocol and fresh test subjects, or a sequential design specified before the original unblinding.

## 14. Compute, storage, annotation, and schedule

The historical log records about 70 minutes for 1,087 questions with 30 sampled answers and about 25 minutes for its feature pass. Those measurements are a reference only. The new two-track experiment adds answer replays, confidence calls, more feature definitions, a second model, and interventions.

For 6,000–9,600 fresh questions, each model produces 54,000–86,400 primary answer sequences: one greedy plus eight sampled answers per question. Confidence collection can require up to the same number of additional calls, with caching of duplicate candidate answers. The two-model plan roughly doubles those counts. Paraphrases, the 733-question check, and causal interventions add work.

Benchmark the 50-question smoke test at realistic batch sizes. Record generated tokens/second, replay tokens/second, questions/hour, peak memory, and disk bytes/question. Estimate the full cost from these measured components, then add a stated contingency. Do not promise completion from the old generation timing alone.

The pilot stored about 5.5 GB of raw tensors for 1,087 questions at two positions. Linear scaling would put an equivalent primary collection at roughly 30–50 GB before additional sampled-answer positions. Full raw storage for every sampled answer could reach hundreds of GB. Instead:

- Store scalar summaries and fixed random projections broadly.
- Store exact raw vectors only for the training-fitted PCA path and a preregistered audit subset, or replay tokens for selected representations.
- Fit streaming PCA on training data only; then replay all splits through the frozen transform.
- Precompute weight-column norms. Compute contribution magnitudes without materializing every vector `c_j`.
- Benchmark literal W2 similarities separately; they add matrix operations across layers.
- Save token IDs and seeds so expensive representations can be reproduced.

Annotation can be the bottleneck. Let U be the number of unique question–candidate pairs and t the measured review time in minutes. Two-reviewer effort is approximately `2 * U * t / 60` hours, plus disagreement resolution. Measure U and t during development. Budget final-label review explicitly rather than assuming automatic matches are ground truth.

A reasonable work sequence is several days for contracts and data assembly, several days for development and power analysis, then measured collection and annotation time. A calendar date should follow the benchmark and annotation sample. No long GPU collection or model download is part of writing this plan.

## 15. Final report and completion checklist

The final report should open with one sentence per subclaim: supported, useful effect ruled out, harmful, or inconclusive. It should include:

- A primary B versus B+M table with AUROC, Brier score, calibration, latency, and paired intervals.
- A direct H versus H+M replication table and M-only replacement result.
- A within-question comparison with eligible-subject counts.
- A capacity/feature ablation table.
- Reliability and risk–coverage plots.
- Concentration-versus-success plots and intervention results, including wrong high-concentration answers.
- A model-by-model replication table and a population/split provenance table.
- A machine-readable decision file applying the registered thresholds.

### Adversarial self-review of this plan

| Dimension | Question | Resolution |
| --- | --- | --- |
| Contribution | Could any positive result just repeat the pilot? | Main comparison removes reference inputs, strengthens baselines, and uses a fresh test |
| Clarity | Does “confidence improves” have a fixed meaning? | Ranking, probability scoring, calibration, and mechanism receive separate decisions |
| Experimental strength | Can the study distinguish no useful gain from low power? | Practical margins, two-sided decisions, development-only power simulation, and explicit inconclusive outcomes |
| Evaluation completeness | Can difficulty, capacity, or one model explain the result? | Same-question tests, matched controls, another model family, and shift checks |
| Method soundness | Does the feature name match actual computation? | Literal W2 similarity is separate from gated selection and contribution concentration |

**Completion criterion:** The study is complete when every registered subclaim has an evidence-backed decision and all required validity checks pass. It is not complete merely because one comparison has a favorable p-value.
