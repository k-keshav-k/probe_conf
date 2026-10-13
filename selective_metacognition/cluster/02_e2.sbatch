#!/bin/bash
# ---------------------------------------------------------------------------
# Selective metacognition E2 — toy arms A-D with reporting training (GPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          selective_metacognition/cluster/02_e2.sbatch
#
# Overrides:
#   SEEDS=5 ROUNDS=3 sbatch ... 02_e2.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=sm-e2
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=02:00:00
#SBATCH --output=sm-e2-%j.out
#SBATCH --error=sm-e2-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

SEEDS="${SEEDS:-3}"
STEPS="${STEPS:-}"
WARM="${WARM:-}"
ROUNDS="${ROUNDS:-}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-$RUN_DIR/selective_metacognition/e2.json}"
ARGS=""
if [ -n "$STEPS" ]; then ARGS="$ARGS --steps $STEPS"; fi
if [ -n "$WARM" ]; then ARGS="$ARGS --warm-start-steps $WARM"; fi
if [ -n "$ROUNDS" ]; then ARGS="$ARGS --rounds $ROUNDS"; fi

mkdir -p "$(dirname "$OUT")"
echo "host: $(hostname)"; nvidia-smi || true
echo "seeds=$SEEDS steps=${STEPS:-default} warm=${WARM:-default} rounds=${ROUNDS:-default} extra=${EXTRA:-none}"
echo "out=$OUT"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m selective_metacognition.cli e2 --seeds $SEEDS $ARGS $EXTRA --out $OUT
"
