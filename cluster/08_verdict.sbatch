#!/bin/bash
# ---------------------------------------------------------------------------
# 08 — Print the section 3.3 verdict from the saved decision files (CPU).
#
# Submitted automatically by cluster/run_confirmatory.sh after evaluation:
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          --export=ALL,MODEL="$PRIMARY_MODEL" cluster/08_verdict.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-verdict
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=s33-verdict-%j.out
#SBATCH --error=s33-verdict-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"

srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  cd $REPO
  python -m section33.cli verdict --run $RUN_DIR --model $MODEL
"
