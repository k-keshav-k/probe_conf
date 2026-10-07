# Evidence audit — 3 October 2026

## Outcome and scope

The saved headline scores reproduce from saved predictions. The original report overstated their meaning. The revised report describes an offline, gold-dependent question-label experiment.

The review covered collection and evaluation code, data, labels, raw tensors, result tables, report assets, and project notes. It also used the installed Qwen implementation and cached 7B configuration. `REPORT.md`, `REPORT.html`, and `REPORT.pdf` present the revised report.

## Verified evidence

| Item | Result | Evidence |
|---|---|---|
| Main sample | 1,087 questions; 1,046 subject IDs | Data JSONL, labels, and Parquet |
| Sample replay | All 32,610 outputs reproduce saved fractions | Historical matcher replay |
| Feature copies | Numeric JSONL and Parquet columns agree | Audit assertions |
| Raw data | 136 batches; expected IDs and shapes; all values finite | Complete main-run raw scan |
| Distribution, norm, and cosine summaries | 730,464 values agree with Parquet | Recalculation from float16 raw copies |
| Follow-up scores | All 22 AUROCs and intervals reproduce | Two extra OOF files and CSV tables |
| Popularity-bin scores | All follow-up within-bin AUROCs reproduce | OOF scores and metadata |
| Split procedure | Recreated folds have no shared train/test subject | StratifiedGroupKFold, seed 0 |
| Signal semantics | Hook tensors and causal positions pass CPU tests | `tests/test_signal_contract.py` |
| Dimensions | 28 layers; hidden 3,584; MLP 18,944; vocabulary 152,064 | Cached configuration |

The audit did not fit new 7B probes or generate new answers. Shared layer-sweep variants agree with the follow-up tables. The saved OOF files have no fold IDs. Thus, split provenance relies partly on the code and run record.

The five gold-dependent summaries received code review. They were not all reconstructed numerically from language-model weights. Other-run ID joins and label ranges pass checks: quick has 40 questions, fs15 has 60, and q7b has 80. All saved probe-table AUROCs and intervals pass range checks.

## Findings and corrections

| Finding | Code or artifact evidence | Correction or remaining work |
|---|---|---|
| Target is question-level sampling success | `generate_correctness`; threshold in `probe_eval.py` | Report the threshold label, not correctness of one answer |
| Only one gold token enters the feature pass | `extract_features`, `ans_ids` | State the leading-space, first-token procedure |
| All answer-position signals use gold input | Hook positions and model input | Mark hidden, raw, and summary signals as oracle at `ans` |
| Five summaries use gold during calculation | Alignment statistics and candidate construction | Include candidate entropy among gold-dependent features |
| Metadata baseline has gold data | `build_metadata_features` | List all nine inputs, including object popularity and alias count |
| Probability probe uses five scalars | `build_prob_features` | Remove “full next-token distribution” description |
| All raw layers are saved | Main NPZ headers and collector | Distinguish storage from five-layer raw probe sweep |
| Gate statistics use pre-activations | Summary calculation | Remove binary-gate and individual-fact interpretations |
| Squared entropy normalizes squares | `entropy_logits_np(log(g_sq + EPS))` | Correct earlier `softmax(g²)` description |
| Final hidden read includes final RMSNorm | Installed Qwen implementation and test | State the special final-layer read |
| Accuracy and positive-label rate differ | Mean fraction versus mean threshold label | Distinguish 17.05% sampled accuracy and 16.47% positives |
| Three confidence parses fail | Saved values −1 | Omit them from descriptive plots; disclose historical probe clipping |
| Substring-only matches affect labels | 56 samples; three threshold changes | Record a stricter diagnostic; retain historical scores pending review |
| Bins do not remove all difficulty | Wide page-view ranges and remaining metadata | Limit claims to ranking within bins |
| Reported residual is not a regression residual | `resid = frac - surface_oof` | Use “diagnostic gap”; disclose target mismatch |
| Observed and bootstrap-mean deltas differ | OOF AUROCs and paired samples | Report observed +0.037 and original interval |
| Method selection uses OOF scores | Best-layer and variant summaries | Require an independent final test set |
| Inner preprocessing sees inner validation data | `evaluate` fits preprocessing before inner folds | Disclose the procedure; require fully nested preprocessing next |
| Bootstrap ignores subject clustering | Question-index resampling | State conditional uncertainty; use subject bootstrap later |
| Confidence and answer prompts differ | Prompt-building functions | Disclose omitted examples and absent candidate answer |
| Raw-row alignment failed for permuted IDs | `align` | Correct mapping; reject duplicate and mismatched ID sets |
| Explainer gate bars ignored the position control | Gate data were always from `ans` | Save and select gate data for both positions |
| Final deployment is not implemented | No persisted classifier or threshold policy | Mark live use and attention pooling as proposed work |

The alignment fix does not change the saved pilot. Feature, label, and raw IDs already had identical order. Historical numeric evidence remains intact. The audit does not alter the old preprocessing procedure or claim corrected-label probe performance.

## Matcher diagnostic

A token-boundary-only replay gives mean fraction correct 0.168752 and 176 positive questions. The historical matcher gives 0.170469 and 179 positives.

| Question | Historical fraction | Token-boundary fraction |
|---|---:|---:|
| Who is the mother of Thor? | 0.633333 | 0.466667 |
| Who is the mother of Napoleon? | 0.500000 | 0.466667 |
| Who is the father of Adad? | 0.500000 | 0.400000 |

The 56 substring-only matches require further review. The stricter matcher is a diagnostic, not an adjudicated replacement. Some joined names can be valid variants. Other matches accept aliases inside unrelated words. Alias review remains necessary. Greedy output text is absent, so greedy labels cannot receive the same independent replay.

## Supporting-file status

- `PLAN.md` records a historical design. It proposes some features and models absent from the completed pilot.
- Meeting files record hypotheses and discussion. They do not establish the difficulty-proxy thesis. The original transcript remains intact.
- `LITERATURE.md` is a research notebook, not a systematic review. Unsupported novelty and raw-linear generalization claims were corrected.
- `EXPERIMENTS.md` records reported run times and hardware measurements. The audit did not repeat those measurements.
- `quick`, `fs15`, and `q7b` artifacts are small pipeline checks. They do not provide the headline result. Their scope and saved tables were inspected; their classifiers were not retrained.
- Original CSV, JSONL, Parquet, and NPZ files retain historical labels and metrics. Report plots and pages now describe them correctly.

## Claim-to-evidence map

| Claim | Evidence | Assessment |
|---|---|---|
| Internal probes rank the saved label better than confidence | Recomputed AUROCs 0.863, 0.871, 0.907 versus 0.532 | Supported for this offline experiment |
| Combined score improves the selected hidden score | Observed delta +0.036571; interval [0.023097, 0.050578] | Supported conditionally; selection remains a limit |
| Internal scores retain ranking within popularity bins | Recomputed within-bin table | Supported; not complete difficulty removal |
| Strongest features require gold data | Supplied gold token and five summaries | Supported by code |
| Oracle-free features detect live mistakes | No matching experiment or persisted probe | Needs a new experiment; not claimed |
| Nonlinear summaries cause the gain or prove a memory theory | No matched ablation or causal intervention | Unsupported; removed |

## Revision outline and self-review

The report order is scope, prior work, data, flow, signal definitions, training, results, limits, and next experiment.

| Dimension | Assessment |
|---|---|
| Contribution | Useful offline comparison; novelty and causal claims removed |
| Writing clarity | Tensor names, formulas, dimensions, and gold dependencies defined |
| Experimental strength | Saved scores reproduce; oracle-free and corrected-label runs still needed |
| Evaluation completeness | Stronger baselines and deployment tests specified as next work |
| Method soundness | Exact preprocessing and bootstrap limits disclosed |

The report uses short sentences, active constructions, and consistent technical nouns based on ASD-STE100 guidance. Full Issue 9 dictionary compliance has not been certified. The [official ASD-STE100 guidance](https://www.asd-ste100.org/about_STE.html) remains the reference.

The citation review verified identifiers, titles, and attribution for the retained report references. It used primary arXiv records, the PopQA page, and the Qwen configuration. It did not independently rederive every theorem claim in the older notebook.

## Repeat the checks

```bash
.venv/bin/python -B scripts/audit_artifacts.py --full-raw
PYTHONPATH=. .venv/bin/python -B -m unittest discover -s tests -v
node tests/check_explainer.js
```

Detailed numerical evidence is in `results/audit_checks.json`. Asset rebuild commands are in the report appendix.
