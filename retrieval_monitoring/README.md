# Retrieval monitoring

Implementation of `report/RETRIEVAL_MONITORING_PLAN.md`: can a model tell when
its factual recall is unreliable? This package is separate from the Section 3.3
confirmatory code in `section33/`.

## Status

| Stage | Status |
| --- | --- |
| S0 repaired text baselines | Implemented. Run on the frozen confirmatory cohort. Result: `report/RETRIEVAL_MONITORING_S0_RESULT.json` |
| S1 retained-state contract | Implemented with the mask fix. Passed on 0.5B and 7B. |
| S2 toy world | Implemented with low-rank relation operators. Passed on 3 seeds. |
| S3 40-subject pilot | Ran once. Gate failed: obscure subjects sat at the success floor and `alpha=0.5` moved the state by ~1% of its norm. Pair fetch now orders by sitelinks and a development sweep (`sweep`) picks layer/rank/alpha before the rerun. |
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
| `sweep` | S3 development sweep over layer, rank, and alpha |
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

# S3: fetch pilot pairs on the login node (needs internet). Famous subjects
# first (sitelinks filter); obscure subjects sit at the success floor.
bash cluster/run.sh python -m retrieval_monitoring.cli fetch-pairs \
  --out "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv" --limit 400

# S3: development sweep over layer, rank, alpha (GPU, ~10-20 min)
bash cluster/run.sh python -m retrieval_monitoring.cli sweep \
  --run "$RUN_DIR" --model "$PRIMARY_MODEL" \
  --source "$RUN_DIR/source/wikidata_cohort.csv" \
  --pairs "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv"

# S3: selectivity pilot (GPU) with the swept settings
bash cluster/run.sh python -m retrieval_monitoring.cli pilot \
  --run "$RUN_DIR" --model "$PRIMARY_MODEL" \
  --source "$RUN_DIR/source/wikidata_cohort.csv" \
  --pairs "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv" \
  --layer 13 --rank 2 --alpha 10
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

## S3 development log

First pilot (Qwen2.5-7B-Instruct, layer 13, rank 2, alpha 0.5, 40 subjects):

- `clean_success` 0.142, floor share 0.725. The pairs fetch returned obscure
  low-QID subjects, so most cells scored 0/8 even clean and damage had no room
  to show.
- `damage_relevant` 0.0, `damage_other` and `damage_random` 0.0016. The
  realized perturbation norm was 0.525 against subject-state norms around 17,
  i.e. about 1 percent. A local 0.5B replication gave a total-variation
  distance of 0.002 between clean and relevant next-token distributions at
  alpha 1; the distance grows to about 0.03 at alpha 10 and 0.13 at alpha 50.
- Confidence parsing was fixed separately: the chat-interface elicitation
  parses 100 percent, the earlier terse plain-text suffix parsed none.

The rerun uses famous subjects and the sweep-selected layer/rank/alpha. The
sweep reports the teacher-forced reference-answer log-probability drop for
relevant, other, and norm-matched random states, plus the realized perturbation
fraction of the state norm.

## What the plan asks next

1. Refetch the pairs CSV (famous subjects), run the S3 sweep, and pick a
   setting with positive relevant damage and clear selectivity over the
   controls.
2. Rerun the 40-subject pilot with those settings. The gate needs
   `damage_relevant >= 0.05` and `selectivity_vs_random >= 0.03` in answer
   success.
3. Only after the selectivity gate: M1 native reporting, M2 learned reporting,
   M3 pathway tests.
