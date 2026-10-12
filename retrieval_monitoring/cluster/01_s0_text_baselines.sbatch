#!/bin/bash
# ---------------------------------------------------------------------------
# Retrieval monitoring S0 — repaired text-only baselines (CPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          retrieval_monitoring/cluster/01_s0_text_baselines.sbatch
#
# Rebuilds both tracks from the frozen cohort manifest and the canonical
# unsharded collection, validates draw counts, and reports the observed gap
# against the saved hidden probe with subject-bootstrap intervals.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rm-s0
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=rm-s0-%j.out
#SBATCH --error=rm-s0-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

TRACKS="${TRACKS:-A B}"

echo "host: $(hostname)"
echo "run root: $RUN_DIR"
echo "tracks: $TRACKS"

srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  python -m retrieval_monitoring.cli text-baselines --run $RUN_DIR --tracks $TRACKS
"
