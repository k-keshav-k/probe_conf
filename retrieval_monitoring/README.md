# Retrieval monitoring

Implementation of `report/RETRIEVAL_MONITORING_PLAN.md`: can a model tell when
its factual recall is unreliable? This package is separate from the Section 3.3
confirmatory code in `section33/`.

## Status

| Stage | Status |
| --- | --- |
| S0 repaired text baselines | Implemented. Run on the frozen confirmatory cohort. Result: `report/RETRIEVAL_MONITORING_S0_RESULT.json` |
| S1 retained-state contract | Implemented with the mask fix. CPU tests pass. The real-model check (`check-state`) needs a GPU node. |
| S2 toy world | Implemented with low-rank relation operators. CPU smoke passes. |
| S3 40-subject pilot | Runner and analysis implemented. Needs a cohort with two questions per subject and a GPU node. |
| M1 / M2 / M3 | Not implemented. These wait on the S3 selectivity gate. |

## Modules

| Module | Purpose |
| --- | --- |
| `retained_state` | Run a prefix once, preserve the mask, fork the KV cache into branches |
| `checks` | S1 apparatus checks on a real model |
| `text_baselines` | S0 baselines with cohort filtering and draw-count validation |
| `interventions` | Relation maps, attenuation, matched random controls |
| `evaluation` | Paired Brier endpoint and the within-question shuffle test |
| `toy_world` | S2 one-layer toy with embedding-level interventions |
| `collector` | Resumable retained-state generation and scoring |
| `pilot` | S3 selectivity analysis and pair selection |
| `runner` | GPU runner for the S3 pilot |
| `cli` | Command line entry points |

## Commands

```bash
# S0: repaired baselines (CPU, saved data)
.venv/bin/python -m retrieval_monitoring.cli text-baselines \
  --run runs/section33_confirmatory --tracks A B

# S1: apparatus checks on a real model (GPU recommended)
.venv/bin/python -m retrieval_monitoring.cli check-state \
  --model Qwen/Qwen2.5-0.5B-Instruct --dtype float32

# S2: toy world (CPU)
.venv/bin/python -m retrieval_monitoring.cli toy --steps 600 --seeds 3

# S3: fetch pilot pairs on the login node (needs internet)
bash cluster/run.sh python -m retrieval_monitoring.cli fetch-pairs \
  --out "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv" --limit 400

# S3: selectivity pilot (GPU)
bash cluster/run.sh python -m retrieval_monitoring.cli pilot \
  --run "$RUN_DIR" --model "$PRIMARY_MODEL" \
  --source "$RUN_DIR/source/wikidata_cohort.csv" \
  --pairs "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv"
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_retrieval_monitoring.py -q
```

## S0 result on the frozen cohort

The repaired baselines use 2,000 training subjects (the manifest size), 800
development subjects, and 4,000 test subjects. Per-question draw counts are
validated: one answer for Track A and eight draws for Track B.

| Track / mode | Baseline AUROC | Saved H | Observed gap (H minus text) | 95% interval |
| --- | --- | --- | --- | --- |
| A / question | 0.789 | 0.888 | 0.098 | [0.075, 0.122] |
| B / question | 0.797 | 0.935 | 0.138 | [0.118, 0.161] |
| B / question + candidate | 0.866 | 0.935 | 0.070 | [0.054, 0.087] |

The hidden probe keeps a clear advantage over these tested TF-IDF baselines.
The question-only within-question AUROC is 0.500 on Track B (no signal), and
the question-plus-candidate baseline reaches 0.663. This is a tested baseline,
not the strongest possible input-only observer.

## What the plan asks next

1. Run the S1 real-model checks on the small model and then 7B, using the exact
   proposed templates and automatic subject spans.
2. Run the S2 toy to debug the intervention if needed.
3. Build a pilot cohort with two questions per subject, then run the S3
   40-subject selectivity pilot.
4. Only after the selectivity gate: M1 native reporting, M2 learned reporting,
   M3 pathway tests.
