#!/bin/bash
# ---------------------------------------------------------------------------
# 06 — Development selection, power simulation, and frozen evaluation (CPU).
#
# Submit from the login node after collection:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          --export=ALL,MODEL="$PRIMARY_MODEL" cluster/06_evaluate.sbatch
#
# Stage 3 (develop) fits the standard probes on the training split and writes
# probes/dev_selection__*.json plus a power table.  Stage 4 (evaluate) fits on
# train, calibrates on the calibration split, evaluates once on the test split
# and writes evaluation/decision__*.json.  Runs both tracks.
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-evaluate
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=s33-evaluate-%j.out
#SBATCH --error=s33-evaluate-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
# Resolve the repository: explicit REPO, the submit directory, or the cwd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"

module purge
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  for TRACK in A B; do
    echo '===== develop Track' \$TRACK '====='
    python -m section33.cli develop --run $RUN_DIR --model $MODEL --track \$TRACK
    echo '===== evaluate Track' \$TRACK '====='
    python -m section33.cli evaluate --run $RUN_DIR --model $MODEL --track \$TRACK
  done
  echo '===== audit ====='
  python -m section33.cli audit --run $RUN_DIR
"
