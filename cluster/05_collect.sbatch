#!/bin/bash
# ---------------------------------------------------------------------------
# 05 — Collect one track and one split, sharded across a Slurm job array.
#
# Submit from the login node (example: Track A, training split, 4 shards):
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-3 \
#          --export=ALL,TRACK=A,SPLIT=train,SHARDS=4,MODEL="$PRIMARY_MODEL" \
#          cluster/05_collect.sbatch
#
# First validation run (small):
#   sbatch ... --array=0-0 --export=ALL,TRACK=A,SPLIT=train,SHARDS=1,LIMIT=8 ...
#
# Each array task is resumable.  To use idle stakeholder GPUs, add
#   --comment="preemption=yes;requeue=true"
# on the sbatch command line.  Preempted tasks requeue and skip finished
# questions.
#
# GPU efficiency: jobs on l40s/h200 nodes are cancelled below 50-60% GPU
# utilization (see the docs).  Keep batches large; do not sleep between calls.
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-collect
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=12:00:00
#SBATCH --output=s33-collect-%A_%a.out
#SBATCH --error=s33-collect-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
# Resolve the repository: explicit REPO, the submit directory, or the cwd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

TRACK="${TRACK:?set TRACK=A or TRACK=B}"
SPLIT="${SPLIT:?set SPLIT=train|dev|calibration|test}"
SHARDS="${SHARDS:-1}"
MODEL="${MODEL:-$PRIMARY_MODEL}"
LIMIT="${LIMIT:-}"

if [ ! -f "$RUN_DIR/manifests/cohort.parquet" ]; then
  echo "ERROR: cohort manifest not found: $RUN_DIR/manifests/cohort.parquet" >&2
  echo "Build the cohort first (cluster/README.md section 6):" >&2
  echo "  sbatch --account=\"\$TORCH_ACCOUNT\" --chdir=\"\$REPO\" ... cluster/04_cohort.sbatch" >&2
  exit 1
fi

if [ "$SHARDS" -gt 1 ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: SHARDS=$SHARDS requires a Slurm job array: --array=0-$((SHARDS - 1))" >&2
  exit 1
fi
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "host: $(hostname)"
echo "track=$TRACK split=$SPLIT shard=$SHARD/$SHARDS model=$MODEL limit=${LIMIT:-all}"
nvidia-smi || true

EXTRA=""
if [ -n "$LIMIT" ]; then EXTRA="--limit $LIMIT"; fi

SHARD_ARG=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SHARD_ARG="--shard $SHARD"; fi

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export TORCH_HOME=$TORCH_HOME
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  python -m section33.cli collect \
    --run $RUN_DIR \
    --model $MODEL \
    --track $TRACK \
    --split $SPLIT \
    --shards $SHARDS \
    $SHARD_ARG \
    --batch 4 \
    $EXTRA
"
