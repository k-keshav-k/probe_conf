# section33 — implementation of the Section 3.3 test plan

This package implements `report/SECTION_3_3_TEST_PLAN.md`. It does not run the
experiment. Run the stages on the cluster, one model at a time.

## What is implemented

| Module | Purpose |
| --- | --- |
| `section33/config.py` | Frozen protocol: margins, splits, prompts, budgets, seeds |
| `section33/schema.py` | Separate row schemas for labels, features, predictions, confidence |
| `section33/features/mlp_math.py` | `a=W_gate x`, `s=SiLU(a)*W_up x`, `o=W_down s`, contribution mass `e_j=|s_j|·||v_j||` |
| `section33/features/families.py` | Frozen families: literal W2 similarities, gate/selection, contribution concentration, output/direction, pilot-12 replication |
| `section33/features/hidden.py` | Seeded fixed projections of pre/post hidden states |
| `section33/features/probability.py` | Probability features P: top-1, margin, entropy, top-k mass, sequence log-probs |
| `section33/models/contract.py` | Hook capture and the bias/reconstruction contract |
| `section33/models/collect.py` | Track A (before-answer) and Track B (after-answer) collection with replay and resume |
| `section33/data/cohort.py` | Fresh cohort, exclusion, equal relation/popularity quotas, subject-grouped splits, hashed manifests |
| `section33/data/labels.py` | Frozen token-boundary matching policy and adjudication helpers |
| `section33/data/wikidata.py` | Wikidata Query Service helper for the four relations |
| `section33/probes/pipeline.py` | Fold-internal imputation/scaling/PCA, B/B+M/H/M specs, 512-dim budget, nonlinear control |
| `section33/probes/calibration.py` | Identity/Platt/isotonic selection, reliability, risk-coverage, corrected calibration error |
| `section33/analysis/difficulty.py` | Train-only strata, within-question AUROC, cross-fitted difficulty models, residuals |
| `section33/analysis/evaluation.py` | Paired subject bootstrap, Holm correction, frozen yes/no rules |
| `section33/analysis/interventions.py` | Identity, top-k attenuation, matched random controls, flatten/sharpen |
| `section33/analysis/power.py` | Subject-level power simulation over the frozen test-size grid |
| `section33/cli.py` | Stage commands |
| `section33/smoke.py` | CPU contract checks and small-model end-to-end smoke |
| `tests/test_section33_contracts.py` | Contract, leakage, calibration, decision, intervention tests |

## Reference-free boundary

Feature files never contain reference answers or labels. The plan requires this
(no reference token, alias, object popularity, or correctness label may enter
the feature extractor). `FeatureRow` has no field for them, and `section33.cli
audit` fails if the strings appear in a feature file.

## Quick checks (safe on a laptop)

```bash
.venv/bin/python -m section33.smoke                 # CPU contracts + pipeline
.venv/bin/python -m section33.smoke --end-to-end    # + Qwen2.5-0.5B, two questions
.venv/bin/python -m pytest tests/test_section33_contracts.py -q
```

## Cluster sequence

```bash
# 0. freeze the run layout and protocol
python -m section33.cli setup

# 1. fresh cohort (Wikidata export or a prepared table)
python -m section33.cli cohort \
    --source data/wikidata_cohort.csv \
    --seen results/pilot7b_labels.csv results/q7b_labels.csv \
           results/fs15_labels.csv results/quick_labels.csv \
    --per-relation 1500

# 2. collection, one model at a time, one track at a time
python -m section33.cli collect --model Qwen/Qwen2.5-7B-Instruct --track A --split train
python -m section33.cli collect --model Qwen/Qwen2.5-7B-Instruct --track A --split dev
# ... calibration, test ...
python -m section33.cli collect --model Qwen/Qwen2.5-7B-Instruct --track B --split train
# ... calibration, test ...

# 3. development: probe selection and power simulation
python -m section33.cli develop --model Qwen/Qwen2.5-7B-Instruct --track A

# 4. frozen evaluation and decision file
python -m section33.cli evaluate --model Qwen/Qwen2.5-7B-Instruct --track A

# 5. schema and reference-exclusion audit
python -m section33.cli audit
```

Repeat steps 2–4 for `mistralai/Mistral-7B-Instruct-v0.3`. The protocol
expects development-only selection, calibration on the calibration split, and
one final evaluation on the test split.

## Stage gates (plan section 13)

- Stage 0: `setup` writes the protocol with a content hash. Do not change it.
- Stage 1: smoke checks and `pytest` must pass before collection.
- Stage 2: `cohort` must report zero overlap with the seen subjects and usable
  class balance. The 733-question legacy check stays separate.
- Stage 3: `develop` writes the selection and the power table. Freeze the test
  size and margins here.
- Stage 4: `evaluate` fits on train, calibrates on calibration, and writes
  `evaluation/decision__*.json` with the registered verdict.
- Stages 5–7: second model, mechanisms, and the written verdict.

## Notes and limits of this implementation

- Hidden-state baselines use fixed random projections (H) for broad storage.
  The exact-vector PCA path requires replaying saved token IDs with the model,
  as the plan permits. Token IDs and seeds are saved for this purpose.
- `Mistral-7B-Instruct-v0.3` uses the same hook contract; `verify_mlp_contract`
  must be run for each new architecture before collection.
- The CLI does not download models or run long jobs automatically.
- The pilot artifacts are not modified. Outputs go to `runs/section33_v1/`.
