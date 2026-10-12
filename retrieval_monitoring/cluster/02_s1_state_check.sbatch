#!/bin/bash
# ---------------------------------------------------------------------------
# Retrieval monitoring S1 — retained-state apparatus check (GPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          retrieval_monitoring/cluster/02_s1_state_check.sbatch
#
# Checks the exact prefix/branch interface on the small model and then 7B:
# clean and left-padded fork-versus-full equivalence, subject-span location,
# a middle-layer intervention that reaches both branches, a last-layer
# intervention that cannot, real multi-token generation, and clean replay.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rm-s1
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=01:00:00
#SBATCH --output=rm-s1-%j.out
#SBATCH --error=rm-s1-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct}"
DTYPE="${DTYPE:-float16}"

echo "host: $(hostname)"; nvidia-smi || true
echo "models: $MODELS"
echo "dtype: $DTYPE"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export TORCH_HOME=$TORCH_HOME
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  for MODEL in $MODELS; do
    echo \"===== state check: \$MODEL =====\"
    python -m retrieval_monitoring.cli check-state --model \$MODEL --dtype $DTYPE
  done
  echo \"===== state check complete =====\"
"
