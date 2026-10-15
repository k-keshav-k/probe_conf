#!/bin/bash
# ---------------------------------------------------------------------------
# P0 — reporter positive control on regenerated toy diagnostic rows.
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          causal_confidence/cluster/01_p0.sbatch
#
# Overrides:
#   ARM=C BATCHES=80 sbatch ... 01_p0.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cc-p0
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=01:00:00
#SBATCH --output=cc-p0-%j.out
#SBATCH --error=cc-p0-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

ARM="${ARM:-B}"
BATCHES="${BATCHES:-40}"
RUN_ID="${RUN_ID:-p0}"
export CAUSAL_RUN_DIR="${CAUSAL_RUN_DIR:-$RUN_DIR/causal_confidence/cc_v1/$RUN_ID}"

echo "host: $(hostname)"; nvidia-smi || true
echo "run dir: $CAUSAL_RUN_DIR"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export OMP_NUM_THREADS=4
  cd $REPO
  python -m causal_confidence.cli setup
  python -m causal_confidence.cli p0 --source toy --arm $ARM --batches $BATCHES
"
