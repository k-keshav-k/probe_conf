#!/bin/bash
# ---------------------------------------------------------------------------
# Knowledge-variant feasibility study on the cluster (one seed per job).
#
# Trains counterbalanced answer variants on inventoried facts, verifies that
# knowledge moved on a held-out paraphrase, trains the shared confidence
# reporter (T1 pointwise vs T2 paired), and evaluates held-out subjects and a
# held-out knowledge adapter.  See report/CAUSAL_CONFIDENCE_TRAINING_PLAN.md
# and the revised protocol in EXPERIMENTS.md.
#
# Submit from the login node:
#   source cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          causal_confidence/cluster/05_knowledge.sbatch
#
# Overrides (pick pilot sizes; the full study is 60/15/36/12):
#   MODEL=Qwen/Qwen2.5-7B-Instruct SEED=0 SUBJECTS=40 EPOCHS=40 \
#     sbatch ... 05_knowledge.sbatch
#
# Notes:
# - Prefer one job at a time; the study is generation-heavy.
# - Outputs: $CAUSAL_RUN_DIR (default $RUN_DIR/causal_confidence/cc_v1/know_seed<SEED>)
#   with variants/, evaluation/, manifests/.  Rerun with the same RUN_ID only
#   after deleting variants/ (checkpoints are reused) or use a new RUN_ID.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cc-know
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=16:00:00
#SBATCH --output=cc-know-%j.out
#SBATCH --error=cc-know-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${SEED:-0}"
SUBJECTS="${SUBJECTS:-40}"
TRANSFER_SUBJECTS="${TRANSFER_SUBJECTS:-10}"
TRAIN_SUBJECTS="${TRAIN_SUBJECTS:-24}"
DEV_SUBJECTS="${DEV_SUBJECTS:-8}"
EPOCHS="${EPOCHS:-40}"
RANK="${RANK:-8}"
TRAIN_DRAWS="${TRAIN_DRAWS:-8}"
EVAL_DRAWS="${EVAL_DRAWS:-16}"
RUN_ID="${RUN_ID:-know_seed${SEED}}"
OUT="${OUT:-$RUN_DIR/causal_confidence/cc_v1/$RUN_ID}"

echo "host: $(hostname)"; nvidia-smi | head -12 || true
echo "run dir: $OUT"
echo "model: $MODEL seed: $SEED subjects: $SUBJECTS epochs: $EPOCHS rank: $RANK"

mkdir -p "$OUT"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m causal_confidence.cli knowledge \
    --seed $SEED \
    --model $MODEL \
    --out $OUT \
    --subjects $SUBJECTS \
    --transfer-subjects $TRANSFER_SUBJECTS \
    --train-subjects $TRAIN_SUBJECTS \
    --dev-subjects $DEV_SUBJECTS \
    --epochs $EPOCHS \
    --rank $RANK \
    --train-draws $TRAIN_DRAWS \
    --eval-draws $EVAL_DRAWS
"
