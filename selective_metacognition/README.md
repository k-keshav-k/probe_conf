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

Overrides: `SEEDS`, `STEPS`, `WARM`, `ROUNDS` (E2), `OUT`, and `EXTRA` for
arbitrary CLI flags. The informative E1 regime:

```bash
EXTRA="--dense --overlap 0.5 --spurious 0.3 --code-dim 128 --d-model 32 --n-layers 1 --lambda-sel 5" \
SEEDS=5 WARM=1000 STEPS=1200 \
  sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  selective_metacognition/cluster/01_e1.sbatch
```

Results go to `$RUN_DIR/selective_metacognition/e1.json` and `e2.json`; logs
are `sm-e1-*.out` and `sm-e2-*.out` in the repo root. Pull the results back
with:

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

### Making E1 informative

With the proposal's disjoint-subspace default, ordinary training already
achieves perfect selectivity (clean 1.000, irrelevant-swap damage 0.000), so
arm C has nothing to fix. The informative regime adds two things:

- **Compression and capacity limits**: dense relation maps, `overlap=0.5`,
  `code_dim=128` into a 256-dim memory, `d_model=32`, one layer.
- **Spurious correlations**: with `spurious > 0`, a fraction of training
  subjects has the same answer in every relation. The unqueried components then
  predict the queried answer, so a clean-trained model can learn to use them.
  Evaluation uses independent bindings, where that shortcut fails.
  Arm C's selectivity loss pairs each binding with one where **all** unqueried
  components are decorrelated, which teaches reliance on the queried component.

Local E1 result (3 seeds, `spurious=0.3`, `lambda_sel=5`, 1000 warm-start and
1200 training steps), saved to
`runs/selective_metacognition/e1_spurious03.json`:

| Arm | clean top-1 (test) | clean top-1 (fresh) | selectivity S |
| --- | --- | --- | --- |
| B | 0.630 | 0.607 | 0.499 |
| C | 0.650 | 0.630 | 0.487 |

Cluster E1 with 5 seeds (`runs/section33_v1/selective_metacognition/e1.json`),
C - B differences:

| Metric (fresh bindings) | Mean | SD | Seeds positive |
| --- | --- | --- | --- |
| clean top-1 | +0.026 | 0.014 | 5/5 |
| valid-swap top-1 | +0.016 | 0.011 | 5/5 |
| irrelevant-swap stability | -0.000 | 0.005 | 3/5 |
| selectivity S | -0.014 | 0.004 | 0/5 |

C resists the spurious shortcut better on independent bindings, consistently
across seeds, without constant-output collapse (`top1_share` 0.03). The S
metric declines because weakening an unqueried component removes the shortcut
that B relies on, which confounds S in this regime. The gain is modest and
could still reflect extra supervision, which is what arm E controls.

## Training arms

| Arm | Retrieval update | Confidence update |
| --- | --- | --- |
| A | none | proper-score reporting |
| B | correctness reward | same reporting training |
| C | B plus the direct selectivity loss | same reporting training |
| D | supervised retrieval plus selectivity loss | same reporting training |
| E | B plus the same cross-entropy terms but no consistency term | same reporting training |

E is the matched-supervision control: it sees the same decorrelated inputs as C
but without the structured invariance term, so C - E isolates the consistency
objective from extra data.

After the shared warm start, every base parameter and the MLP down projections
are frozen. Only the gate/up adapters in `adapter_layers` train during
retrieval updates; only the report head trains during reporting updates.

The toy's answer space is small, so the correctness reward is available in
closed form, `E[Y] = pi[a*]`. The primary arms use that exact expected reward
(same expectation as the sampled policy gradient, no estimator variance); the
sampled leave-one-out path is kept in `training.sampled_reward` for an
optional check.

The report head is a distribution over a fixed grid (`0.00, 0.05, ..., 1.00`)
produced by a two-layer adapter (`d_report=256`). It is trained against
outcomes sampled from the frozen answer policy using the risk form of the
proper score: the expected report is trained by squared error against the
empirical outcome mean, on a stored rollout batch (the proposal's rollout
store). Three implementation choices mattered:

- A single linear report head could not compute the softmax-like success
  probability from the hidden state; a deeper adapter tracks condition changes.
- Training online on fresh samples recovered only a few percent of the
  condition effect; training offline on a stored rollout batch recovered about
  a quarter and raised within-question sign agreement to ~0.65.
- The Brier helper double-expanded columnar probabilities, corrupting the
  baseline and augmented scores. Fixed; an informative synthetic report now
  yields a positive incremental Brier.

E2 outcome (cluster, 5 seeds, with the fixed reporting stack).  The original
table used a **privileged baseline** that received the intervention identity,
which the reporter never sees.  The corrected evaluation separates:

- an **ordinary baseline**: relation, max probability, entropy, top-2 margin;
- a **privileged baseline**: the same plus the condition one-hot;
- constant-confidence (0.5 and train success rate) and report-recalibration
  baselines;
- observable-reliability conditions versus the coherent false memory.

Corrected result (cluster, 5 seeds, observable conditions, test):

| Arm | greedy | sampled | report | const 0.5 | const rate | recalibrated | ordinary incr. | privileged incr. | sq. prob. error | sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.588 | 0.408 | 0.267 | 0.250 | 0.231 | 0.225 | +0.00049 | +0.00004 | 0.139 | 0.596 |
| B | 0.617 | 0.439 | 0.276 | 0.250 | 0.238 | 0.230 | +0.00033 | +0.00001 | 0.147 | 0.600 |
| C | 0.641 | 0.421 | 0.268 | 0.250 | 0.234 | 0.227 | +0.00038 | −0.00002 | 0.128 | 0.599 |
| D | 0.638 | 0.419 | 0.269 | 0.250 | 0.233 | 0.227 | +0.00034 | +0.00001 | 0.128 | 0.598 |
| E | 0.640 | 0.438 | 0.275 | 0.250 | 0.237 | 0.230 | +0.00049 | +0.00004 | 0.139 | 0.599 |

Corrected reading:

- **Greedy and sampled success diverge.** C improves greedy top-1 over B
  (+0.024) but has lower sampled success (−0.018), and C matches E on both
  (0.641 vs 0.640 greedy). The accuracy benefit specifically from the
  consistency term is not established.
- **The reporter is weak.** Raw Brier (0.267–0.276) is worse than an always-0.5
  predictor (0.250) and around the train-rate constant (0.231–0.238).
  Recalibration improves it to 0.225–0.230, only slightly better than the
  constant-rate baseline, so its ranking carries a little information but its
  scale is off.
- **Incremental value is negligible.** Against the ordinary baseline the
  increment is +0.0003 to +0.0005 for every arm; C − B is +0.00004 (4/5 seeds)
  and C − E is −0.00012 (2/5). The privileged baseline gives ~0.
- **C and D predict the true success probability better.** Squared probability
  error: C/D 0.128 versus B 0.147 and E 0.139; C − B −0.019 (5/5 seeds) and
  C − E −0.011 (5/5). This is a descriptive finding, not a calibration-only
  metric, and it is the one consistent effect of the consistency objective.
- **Within-question tracking is condition-dependent.** For target weakening,
  sign agreement 0.70 (delta p +0.377, delta c +0.105); for irrelevant
  weakening 0.58; for irrelevant swap 0.52 (effect size ~0.015).
- **Coherent false memories remain confident.** The report predicts ~0.19–0.20
  success while actual success is ~0.007. All arms show this; the boundary is
  reported separately and is not part of the main endpoint.

Defensible verdict: **C improves greedy accuracy over B and reduces the
reporter's squared probability error, but does not demonstrate incremental
reporting value under the current comparison. The reason for that failure
remains unresolved** (reporter learning, redundant features, privileged
baseline information, or evaluation limits). The earlier claim that the toy is
structurally unable to test added information was wrong: sampling does not make
answer probabilities sufficient, because evaluating `pi(a* | q)` requires
knowing which answer is correct.

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

Implemented and tested: world semantics (including `irrelevant_swap_all` and
`swap_scale`), adapters and gradient scopes, warm start with per-step binding
resampling, arms A-E, the rollout-store reporting trainer, E1/E2 runners, CLI
flags for the informative regime, and Slurm jobs.

E1 is confirmed on the cluster (5 seeds): C improves greedy accuracy over B on
fresh bindings, but E (matched supervision) matches C in E2, so the gain
specifically from the consistency term is not established.

E2 audit completed: the ordinary baseline no longer receives intervention
identity; a privileged version is reported separately; constant, recalibrated,
and stronger probability-feature baselines are included; greedy accuracy and
sampled success are reported separately; coherent false memories are evaluated
separately.

Current verdict: C improves greedy accuracy and reduces the reporter's squared
probability error, but does not demonstrate incremental reporting value under
the current comparison. The reason remains unresolved. The next step, if
pursued, is a prospective extension: predict success after a specified change,
with both predictors receiving the same intervention description and neither
seeing the post-intervention state.
