# Section 3.3 follow-up review

Reviewed on 8 October 2026 at commit `1aabbdf`.

**Verdict: substantially improved, but not ready for a confirmatory conclusion.** Several important repairs work. Other findings remain partly open, and the revised join introduces a duplicate-confidence problem. The saved scientific results have not been regenerated with this code.

This review inspected the changes since the previous review, ran the relevant tests and reproduction checks, and performed additional CPU checks. It did not change production code, collect model outputs, or refit the real experiment.

## Confirmed repairs

- **F1, probability selection:** Track A now selects six probability features. Track B selects eleven answer-position probability features. Required empty blocks raise an error.
- **F2, prediction-time boundary:** Track A metadata excludes answer length and confidence missingness. Changing generated-answer length leaves the metadata matrix unchanged.
- **F3, original sample-loss example:** two candidates with identical text and different token sequences now retain all eight draws. Stable candidate keys and unmatched-row checks were added. Other join problems remain below.
- **F4, multiplicity:** both development and evaluation expand multiplicity before fitting, calibration, and scoring, then reset it to one to prevent double expansion.
- **F5, original power failures:** pooled AUROC bootstrap now works with one answer per subject. Sampling uncertainty scales with sample size. The large-sample check with calibration forced to pass now reaches power 1.0.
- **F8, simultaneous AUROC intervals:** secondary and within-question AUROC intervals now use the declared conservative family confidence. The previous ordinary-interval equivalence error is repaired.
- **F9, calibration variance:** the correction now uses centered cluster totals and sizes. Bootstrap copies receive distinct cluster IDs.
- **Parts of F10/F12:** flatten/sharpen now uses contribution mass and passes the rescaling test; partial attenuation mass accounting is repaired. Track A prompt hashes use the corresponding question. Prompt-manifest checks were added.

## Remaining findings

### R1 — High: the confidence join can multiply observations

Location: `section33/datasets.py:120`.

The duplicate check runs before the confidence merge. Confidence rows are neither deduplicated nor joined with a cardinality constraint. Repeated rows from overlapping shards therefore multiply the merged observations. Track A has no later sample-count check to catch this.

**Reproduced:** one label row and one feature row, with two identical confidence rows, return two Track A observations. The existing Track A confidence files contain eight duplicate `(question_id, track, answer_index)` keys, so this is relevant to the saved collection.

The label/feature join also no longer includes subject and split in its key, and does not compare the suffixed metadata. A feature marked `train` joined to a label marked `test` without an error in the reproduction. Deduplication keeps the first conflicting record rather than checking consistency. Stable candidate keys contain model, revision, and answer tokens, but not the run or prompt identity.

**Required:** validate consistent subject, split, prompt/run provenance, multiplicity, and feature version; reject conflicting duplicates; allow identical duplicates only through explicit deduplication; require one-to-one confidence joins. Check cardinality after every merge. Add tests for overlapping shards and conflicting metadata.

### R2 — High: within-question power resampling loses repeated bootstrap copies

Location: `section33/analysis/power.py:81`.

The pooled bootstrap correctly repeats answer rows, but retains the original question IDs. `within_question_auroc()` groups by unique question ID. Drawing one subject twice therefore changes its row count but not its weight in the mean of per-question AUROCs. The required bootstrap must count the two draws separately.

**Reproduced:** for three mixed-outcome subjects with known AUROC differences, 202 of 300 draws differ from the correct bootstrap. The estimated standard deviation is 0.4420 instead of 0.4923 in this example. This can distort the required test size.

There are two additional power gaps. `within_question_power_table()` is called at its default 95% confidence, while final evaluation uses 99.1667% confidence. Its simulator forces the Brier guardrail to pass by supplying `(-1, 0)`. Primary power still assumes a fixed calibration-pass probability rather than simulating the full joint decision; this limitation is now documented, but the full planned power check is not implemented.

**Required:** give each sampled subject/question copy a distinct bootstrap identity, or bootstrap the per-question statistics directly. Use the final corrected confidence level and simulate the joint guardrail. Validate against known resampling cases and freeze the selected test size before evaluation.

### R3 — High: final evaluation does not enforce a confirmatory design

Locations: `section33/cli.py:436` and `section33/data/cohort.py:293`.

`evaluate` verifies protocol and prompt files, then checks for nonempty train/calibration/test data. It does not require a passing confirmatory cohort manifest, a selected powered sample size, the frozen development selection, reviewed labels, or a prior Track A decision before issuing Track B claims.

The new cohort validator checks requested counts, relation presence/balance, and a nonzero exclusion count. It does not establish that the full required exclusion list was applied, enforce power, or preserve the declared target balance within splits. `--allow-seen` can still be combined with confirmatory mode. The default mode is pilot, but evaluation does not distinguish the resulting claims.

**Reproduced:** a cohort with four subjects in each split and an exclusion count of one passes confirmatory validation. The protocol gate passes after `setup` with no cohort manifest or power file. The existing end-to-end test also completes evaluation without a cohort manifest or development/power artifacts.

**Required:** make pilot and confirmatory evaluation explicit. Confirmatory evaluation must verify the cohort, exclusions, split integrity, power/sample-size decision, label review, frozen choices, and claim hierarchy. Pilot evaluation should remain available with an exploratory verdict. Do not infer complete exclusion from `len(seen) > 0`.

### R4 — High: controls are fitted but do not resolve the comparison

Locations: `section33/cli.py:459`, `section33/cli.py:530`, and `section33/probes/pipeline.py:294`.

The matched-budget, larger-hidden, and nonlinear controls now enter the fitting loop. However, they receive only per-method AUROC output. They do not receive the paired Brier/AUROC comparisons and intervals needed to support or constrain the claim decisions.

The old mismatches also remain: hidden projections cover eight layers, while MLP summaries cover all layers; `H_512` is still compared with `H_all_plus_M`, whose hidden block has no matching PCA limit. The B/B+M specifications remain unrestricted. Adding other fitted methods does not repair these comparisons. The nonlinear control runs one seed, not the planned five-seed treatment, and there is no matched nonlinear augmented comparator.

**Required:** define and evaluate the actual matched-budget and additive contrasts, keep the same hidden representation on each side, match layer coverage, and report the planned uncertainty and seed sensitivity. Connect those results to the allowed claims.

### R5 — Medium: saved fitted artifacts cannot reproduce predictions

Location: `section33/cli.py:372`.

`_fitted_artifact()` saves coefficients for a linear classifier and calibration parameters. It omits feature order, imputation state, scaling state, and PCA transforms. It does not save neural-network weights. A `fitted__*.json` file is therefore not a reloadable fitted pipeline.

**Required:** persist the complete fitted estimator and preprocessing with its feature manifest, calibration map, versions, and hashes. Add a save/load prediction-equivalence test for both linear and nonlinear pipelines. Current tests only require that a fitted file exists.

Resume recovery is also incomplete: it checks that a question occurs in all three interfaces, not that all candidate rows, positions, and multiplicities are present. A partially appended feature file can leave a question marked complete. Verify record-level completeness before skipping a question.

### R6 — Medium: the difficulty report does not test added information after difficulty control

Location: `section33/cli.py:396`.

The new report adds useful descriptive reliability and coverage summaries. It fits a linear metadata model and divides test predictions at their test median. This does not implement the planned strong difficulty comparator, frozen strata, cross-fitted difficulty residual analysis, or paired B versus B+M comparison within difficulty groups.

The reported residual correlation is `corr(Y - augmented_score, difficulty_score)`. That is a calibration residual diagnostic. It is not the proposed `corr(Y - difficulty_score, MLP_score)` test of information left after difficulty prediction. These answer different questions. Also, larger predicted correctness is labeled “high difficulty,” reversing the ordinary interpretation; Track B metadata can contain answer length despite the report note claiming pre-answer-only metadata.

**Required:** implement the declared conditional comparison, retain descriptive checks with accurate names, and avoid treating either raw correlation as proof of difficulty independence.

### R7 — Scope gap: the dominance mechanism still has no executed experiment

Location: `section33/analysis/interventions.py` and the command parser in `section33/cli.py:666`.

The invariant intervention math is repaired. There is still no runner that applies these interventions during model generation, collects outcomes on the separate causal cohort, compares against matched random controls, or tests the predicted concentration direction. No such outcome artifact is present.

This is not required to report a narrowly scoped predictive improvement. It is required before claiming the dominance explanation in section 3.3 is supported. Separate literal-W2 and actual-contribution results are also still needed for those mechanistic claims.

### R8 — Robustness: no eligible within-question data causes an exception

Location: `section33/analysis/evaluation.py:367`.

If every question has only correct answers or only incorrect answers, the within-question bootstrap calls `np.concatenate([])` and raises `ValueError`. This is an insufficient-evidence case and should yield an explicit inconclusive component with zero eligible questions. The failure was reproduced.

The within-question Brier interval is still taken from the ordinary 95% primary interval, rather than the corrected secondary procedure. Apply the declared joint secondary rule consistently.

## Verification and current results

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  -q -p no:cacheprovider \
  tests/test_section33_contracts.py tests/test_signal_contract.py
```

Result: **39 passed in 77.71 seconds**. The repository's post-repair reproduction script also passed its original examples. Additional evidence is saved in `report/SECTION33_FOLLOWUP_EVIDENCE.json`; its reproduction script is `report/section33_followup_checks.py`.

The only saved scientific decision remains the October 7 Qwen Track B run with 50 test subjects and an inconclusive verdict. There is no new Track A decision or Mistral replication. Its protocol artifacts predate these repairs and have no prompt manifest. The old results cannot be reinterpreted as results of the corrected implementation.

**Next gate:** repair R1–R4 before treating a run as confirmatory; complete persistence and the intended difficulty test; retain an explicit boundary around the optional causal claim. Then rerun development and freeze the fresh final experiment. Passing code checks alone will not establish that MLP features improve confidence.
