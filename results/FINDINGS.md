# Pilot findings — evidence review of 3 October 2026

The saved scores reproduce. Their interpretation requires gold dependencies and evaluation limits. The complete procedure is in `../report/REPORT.md`. The audit record is in `../report/AUDIT.md`.

## Data and target

The main run uses Qwen2.5-7B-Instruct and 1,087 questions from four PopQA relations. The data contain 1,046 subject IDs. Each question has 30 sampled outputs. The main label is 1 when at least 15 outputs pass the historical matcher.

Mean fraction correct is 0.170469. The main-label positive rate is 179/1,087, or 16.47%. The greedy positive rate is 210/1,087, or 19.32%. These are different quantities.

## Saved main-target results

| Exact feature set | AUROC | 95% question-bootstrap interval |
|---|---:|---|
| Confidence logistic probe | 0.532 | [0.491, 0.573] |
| Nine metadata features | 0.786 | [0.749, 0.820] |
| Gold first-token probability | 0.801 | [0.762, 0.837] |
| Five probability features | 0.885 | [0.859, 0.907] |
| MLP summaries, last prompt, layer 2 | 0.863 | [0.829, 0.891] |
| Hidden state, gold-token position, layer 18 | 0.871 | [0.841, 0.897] |
| Hidden layer 18 + all summaries | 0.907 | [0.882, 0.929] |
| Raw selection activation, gold-token position, layer 16, PCA-256 | 0.853 | [0.822, 0.882] |

The observed combined-versus-hidden difference is +0.036571. Its interval is [0.023097, 0.050578]. Earlier summaries used the bootstrap mean +0.036346. The observed MLP-only difference is −0.007494, with interval [−0.040913, 0.023639]. An interval containing zero does not prove equivalence.

For the greedy target, the fixed combined set gives AUROC 0.884737. Its observed increase over the fixed hidden set is +0.032084. The interval is [0.020762, 0.043955]. The best compact MLP variant differs between targets.

## Signal dependencies

The feature pass uses the answer prompt and one supplied gold-answer token. All gold-token-position activations depend on this input. At the last prompt position, five of the 17 summaries still use gold during calculation. These include candidate-token entropy. No result is available for the remaining 12 features alone.

The metadata baseline includes object popularity and alias count. The probability baseline includes gold first-token probability and rank. Thus, neither baseline is entirely gold-free. The strongest internal comparison is also gold-dependent.

Raw activations are saved at every layer and both positions. Five layers enter the raw-vector probe sweep. The raw probe reads activations, not weight matrices. This comparison does not prove that nonlinear features cause the gain.

## Difficulty diagnostics

The probes retain AUROC within popularity quintiles. The combined probe ranges from 0.855 to 0.922. The selected last-prompt MLP probe ranges from 0.786 to 0.890. The confidence probe ranges from 0.474 to 0.590.

Each bin still contains popularity variation and other difficulty factors. The scores come from global out-of-fold probes. The evaluator does not train separate probes within bins.

The diagnostic gap is `fraction_correct − surface_OOF`. The surface score predicts a binary label. It is not a regression estimate of the continuous fraction. Gap correlations are therefore descriptive. They do not prove removal of difficulty.

## Data and evaluation limits

Three confidence parses failed. The historical confidence probe clips −1 to 0. The descriptive plots now omit failed values. Mean valid confidence is 91.12, over 1,084 questions.

The substring fallback accepts 56 samples beyond token-boundary matches. A stricter diagnostic changes three threshold labels and gives 176 positives. This diagnostic does not replace alias adjudication. The saved scores still use historical labels.

Best layers and variants were selected on pooled out-of-fold scores. Inner preprocessing is fitted before inner validation splitting. Bootstrap samples use questions, not subject groups. The intervals do not include classifier retraining or method-selection uncertainty.

The audit corrected ID alignment for permuted raw rows. The saved pilot already had identical row order, so its scores are unchanged. Three CPU regression tests pass. All 136 main raw batches have correct shapes and finite values.

## Supported conclusion and next work

Internal signals predict the historical question label better than the stated-confidence baseline in this offline experiment. The combined set improves the selected hidden score. This result supports another experiment.

The next experiment must review aliases, remove gold dependencies, match prompt contexts, and use fully nested preprocessing. It must reserve a final test set. Live use also requires a persisted classifier and separate calibration or abstention evaluation.

The proposed cross-layer attention model remains unimplemented. The pilot does not establish a causal memory theory or a confidence signal independent of all difficulty.
