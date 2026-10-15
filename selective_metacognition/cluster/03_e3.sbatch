#!/bin/bash
# ---------------------------------------------------------------------------
# Selective metacognition E3 — prospective reporting (GPU).
#
# The reporter sees the clean state and a description of an intervention that
# will be applied after the report; it predicts success under the
# post-intervention policy.  The baseline gets the clean output-probability
# features and the same intervention description, but not the reporter.
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          selective_metacognition/cluster/03_e3.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=sm-e3
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=02:00:00
#SBATCH --output=sm-e3-%j.out
#SBATCH --error=sm-e3-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

SEEDS="${SEEDS:-5}"
STEPS="${STEPS:-}"
WARM="${WARM:-}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-$RUN_DIR/selective_metacognition/e3.json}"
ARGS=""
if [ -n "$STEPS" ]; then ARGS="$ARGS --steps $STEPS"; fi
if [ -n "$WARM" ]; then ARGS="$ARGS --warm-start-steps $WARM"; fi

mkdir -p "$(dirname "$OUT")"
echo "host: $(hostname)"; nvidia-smi || true
echo "seeds=$SEEDS steps=${STEPS:-default} warm=${WARM:-default} extra=${EXTRA:-none}"
echo "out=$OUT"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m selective_metacognition.cli e3 --seeds $SEEDS $ARGS $EXTRA --out $OUT
"
