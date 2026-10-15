#!/bin/bash
# ---------------------------------------------------------------------------
# P2 discovery — bounded candidate search; freeze one manipulation.
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          causal_confidence/cluster/03_p2_discover.sbatch
#
# Overrides:
#   DRAWS=8 ALLOW_REDUCED=1 sbatch ... 03_p2_discover.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cc-p2d
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=08:00:00
#SBATCH --output=cc-p2d-%j.out
#SBATCH --error=cc-p2d-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

DRAWS="${DRAWS:-}"
RUN_ID="${RUN_ID:-p2}"
export CAUSAL_RUN_DIR="${CAUSAL_RUN_DIR:-$RUN_DIR/causal_confidence/cc_v1/$RUN_ID}"
EXTRA=""
if [ -n "$DRAWS" ]; then EXTRA="$EXTRA --draws $DRAWS"; fi
if [ "${ALLOW_REDUCED:-0}" = "1" ]; then EXTRA="$EXTRA --allow-reduced"; fi

echo "host: $(hostname)"; nvidia-smi || true
echo "run dir: $CAUSAL_RUN_DIR draws=${DRAWS:-default} extra=$EXTRA"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m causal_confidence.cli p2-discover $EXTRA
"
