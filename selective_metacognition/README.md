# Selective MLP retrieval for metacognitive confidence (toy)

Implementation of the staged toy experiments in
`report/SELECTIVE_MLP_METACOGNITION_PROPOSAL.md`. This package is separate from
the failed S3 relation-map runner and does not reuse its intervention.

The core question is whether training a *functionally selective* retrieval
computation makes the model's own confidence report better at predicting its
factual success. The toy world provides exact ground truth for every condition,
so the first comparison (E1) isolates retrieval training before any reporting
claim is made.

## Package layout

| Module | Purpose |
| --- | --- |
| `config.py` | Toy configuration and the condition lists |
| `world.py` | Controlled world `h_s = sum_r T_r v_{a(s,r)}` with known conditions |
| `model.py` | Small gated-MLP decoder, gate/up adapters, report head |
| `training.py` | Warm start, arms B/C/D, reporting updates, alternating procedure |
| `evaluation.py` | Functional-selectivity table and confidence endpoints |
| `runner.py` | E1 and E2 runners with per-seed aggregation |
| `cli.py` | Command line entry points |

## Commands

```bash
# E1: arms B (correctness) and C (correctness + selectivity), three seeds
.venv/bin/python -m selective_metacognition.cli e1 --seeds 3 \
  --out runs/selective_metacognition/e1.json

# E2: arms A-D with alternating reporting training
.venv/bin/python -m selective_metacognition.cli e2 --seeds 3 \
  --out runs/selective_metacognition/e2.json
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_selective_metacognition.py -q
```

## Slurm (NYU Torch)

The package auto-detects CUDA, so the same commands run on a GPU node. On the
login node:

```bash
cd /scratch/kk6081/probe_conf
git pull --ff-only
source cluster/env.sh

# E1: arms B and C, three seeds
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  selective_metacognition/cluster/01_e1.sbatch

# E2: arms A-D with alternating reporting training
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  selective_metacognition/cluster/02_e2.sbatch
```

Overrides: `SEEDS`, `STEPS`, `WARM`, `ROUNDS` (E2), `OUT`. Results go to
`$RUN_DIR/selective_metacognition/e1.json` and `e2.json`; logs are `sm-e1-*.out`
and `sm-e2-*.out` in the repo root. Pull the results back with:

```bash
SRC="kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/selective_metacognition/" \
DEST="runs/section33_v1/selective_metacognition/" \
  bash cluster/sync_from_torch.sh
```

## World and conditions

Each subject is a sum of relation components. Answer codes are unit vectors;
each relation owns a block of the memory, so the selector can read the queried
component without interference. Conditions change known components:

| Condition | Change | Correct answer |
| --- | --- | --- |
| `clean` | none | original |
| `irrelevant_swap` | replace an unqueried component | original |
| `target_weaken` | scale the queried component by `weaken_strength` | original |
| `irrelevant_weaken` | scale an unqueried component | original |
| `target_swap_valid` | replace the queried component | new answer |
| `false_memory` | replace the queried component | original |
| `sham` | random nuisance change with matched norm | original |

Training **resamples the subject bindings every step**. A fixed table lets the
model memorize memory-to-answer mappings; the first run did exactly that
(train top-1 1.000, held-out 0.379). Resampling forces the model to learn the
selection mechanism.

## Training arms

| Arm | Retrieval update | Confidence update |
| --- | --- | --- |
| A | none | proper-score reporting |
| B | correctness reward | same reporting training |
| C | B plus the direct selectivity loss | same reporting training |
| D | supervised retrieval plus selectivity loss | same reporting training |

After the shared warm start, every base parameter and the MLP down projections
are frozen. Only the gate/up adapters in `adapter_layers` train during
retrieval updates; only the report head trains during reporting updates.

The toy's answer space is small, so the correctness reward is available in
closed form, `E[Y] = pi[a*]`. The primary arms use that exact expected reward
(same expectation as the sampled policy gradient, no estimator variance); the
sampled leave-one-out path is kept in `training.sampled_reward` for an
optional check.

The report head is a distribution over a fixed grid (`0.00, 0.05, ..., 1.00`).
It is trained against outcomes sampled from the frozen answer policy, using the
exact expected Brier score against the empirical outcome mean.

## Endpoints

E1 (`functional_table`): clean success/top-1, irrelevant-swap stability,
target-weakening loss, irrelevant-weakening loss, selectivity
`S = p_irrelevant_weaken - p_target_weaken`, valid-swap accuracy,
false-memory accuracy, and constant-output checks (`top1_share`,
`marginal_entropy`).

E2 (`reporting_metrics`): Brier of the report, Brier of a metadata plus
output-probability baseline, incremental Brier from adding the report, the
unbiased squared-probability-error estimate of proposal section 11.2, and
same-question condition tracking (sign agreement and correlation between
success changes and report changes).

## Status

Implemented and tested: world semantics, adapters and gradient scopes, warm
start, arms B/C/D, reporting updates, E1/E2 runners, and CPU tests. Results are
reported separately once the runs complete; the proposal's decision gates
(G0-G5) still apply.
