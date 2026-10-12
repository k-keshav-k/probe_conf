#!/bin/bash
# ---------------------------------------------------------------------------
# Retrieval monitoring S2 — toy-world intervention check (CPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          retrieval_monitoring/cluster/03_s2_toy.sbatch
#
# Trains a one-layer toy with low-rank relation operators and checks factual
# competence, selective damage from the queried relation map against other-map
# and matched-random controls, and probe readability.  Three seeds.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rm-s2
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=rm-s2-%j.out
#SBATCH --error=rm-s2-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

ENTITIES="${ENTITIES:-512}"
DIM="${DIM:-64}"
STEPS="${STEPS:-800}"
SEEDS="${SEEDS:-3}"
OUT="${OUT:-$RUN_DIR/retrieval_monitoring/toy_results.json}"

mkdir -p "$RUN_DIR/retrieval_monitoring"
echo "host: $(hostname)"
echo "toy: entities=$ENTITIES dim=$DIM steps=$STEPS seeds=$SEEDS"
echo "out: $OUT"

srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export PYTHONUNBUFFERED=1
  cd $REPO
  python -m retrieval_monitoring.cli toy \
    --entities $ENTITIES --dim $DIM --steps $STEPS --seeds $SEEDS --out $OUT
"
