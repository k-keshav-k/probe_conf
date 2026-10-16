#!/bin/bash
# ---------------------------------------------------------------------------
# Native confidence arms A/B/C on the verified knowledge variants.
#
# Trains the reporter adapter to emit an actual grid number, optimizing the
# proper score against independently graded answers, with arm A scoring only
# the report suffix, arm B sharing the adapter with question processing (plus
# an answer-preservation term), and arm C as the difficulty control trained on
# cross-variant success.
#
#   source cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          causal_confidence/cluster/06_native.sbatch
#
# Overrides: MODEL, SEED, ROOT (knowledge run dir with variants/), STEPS.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cc-native
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=16:00:00
#SBATCH --output=cc-native-%j.out
#SBATCH --error=cc-native-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1500}"
WARMUP="${WARMUP:-300}"
BETA="${BETA:-0.1}"
ROOT="${ROOT:-$RUN_DIR/causal_confidence/cc_v1/know7b_seed0}"
OUT="${OUT:-$RUN_DIR/causal_confidence/cc_v1/native7b_seed0/native_arms.json}"

echo "host: $(hostname)"; nvidia-smi | head -12 || true
echo "root: $ROOT"
echo "out:  $OUT"
echo "model: $MODEL steps: $STEPS warmup: $WARMUP beta: $BETA"

mkdir -p "$(dirname "$OUT")"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -u scripts/run_native_arms.py \
    --model $MODEL \
    --root $ROOT \
    --out $OUT \
    --steps $STEPS \
    --warmup-steps $WARMUP \
    --beta-answer $BETA \
    --seed $SEED
"
