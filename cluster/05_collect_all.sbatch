#!/bin/bash
# ---------------------------------------------------------------------------
# 05b — collect every track and split in ONE resumable GPU job (no sharding).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          --export=ALL,MODEL="$PRIMARY_MODEL" cluster/05_collect_all.sbatch
#
# There is exactly one writer per output file, so the shard-overlap class of
# errors cannot occur.  Every collect invocation is resumable: if the job is
# preempted or hits the wall time, resubmit the same command and completed
# questions are skipped.
#
# Env knobs: TRACKS (default "A B"), SPLITS (default all four), BATCH (4).
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-collect-all
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=24:00:00
#SBATCH --output=s33-collect-all-%j.out
#SBATCH --error=s33-collect-all-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"
TRACKS="${TRACKS:-A B}"
SPLITS="${SPLITS:-train dev calibration test}"
BATCH="${BATCH:-4}"

if [ ! -f "$RUN_DIR/manifests/cohort.parquet" ]; then
  echo "ERROR: cohort manifest not found: $RUN_DIR/manifests/cohort.parquet" >&2
  echo "Run cluster/04_cohort.sbatch first." >&2
  exit 1
fi

echo "host: $(hostname)"; nvidia-smi || true
echo "model=$MODEL tracks=$TRACKS splits=$SPLITS batch=$BATCH"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export TORCH_HOME=$TORCH_HOME
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  for TRACK in $TRACKS; do
    for SPLIT in $SPLITS; do
      echo \"===== collect track \$TRACK split \$SPLIT =====\"
      python -m section33.cli collect \
        --run $RUN_DIR \
        --model $MODEL \
        --track \$TRACK \
        --split \$SPLIT \
        --shards 1 \
        --batch $BATCH
    done
  done
  echo \"===== collection complete =====\"
"
