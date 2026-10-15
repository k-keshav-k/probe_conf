#!/bin/bash
# ---------------------------------------------------------------------------
# P1 — episode contract, branch isolation, adapter gating, answer audit.
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          causal_confidence/cluster/02_p1.sbatch
#
# Overrides:
#   DRAWS=8 ALLOW_REDUCED=1 sbatch ... 02_p1.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cc-p1
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=03:00:00
#SBATCH --output=cc-p1-%j.out
#SBATCH --error=cc-p1-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

DRAWS="${DRAWS:-8}"
RUN_ID="${RUN_ID:-p1}"
export CAUSAL_RUN_DIR="${CAUSAL_RUN_DIR:-$RUN_DIR/causal_confidence/cc_v1/$RUN_ID}"
EXTRA=""
if [ "${ALLOW_REDUCED:-0}" = "1" ]; then EXTRA="--allow-reduced"; fi

echo "host: $(hostname)"; nvidia-smi || true
echo "run dir: $CAUSAL_RUN_DIR draws=$DRAWS"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m causal_confidence.cli setup
  python -m causal_confidence.cli p1 --draws $DRAWS $EXTRA
"
