#!/bin/bash
# ---------------------------------------------------------------------------
# 07 — Contract gate for the real model on a GPU node.
#
# Submit before the first long collection:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          --export=ALL,MODEL="$PRIMARY_MODEL" cluster/07_gpu_contract.sbatch
#
# Checks biases, the W_down s reconstruction, reference-free feature extraction,
# and prints the GPU type and torch version.
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-contract
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=00:30:00
#SBATCH --output=s33-contract-%j.out
#SBATCH --error=s33-contract-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
# Resolve the repository: explicit REPO, the submit directory, or the cwd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"
echo "host: $(hostname)"; nvidia-smi || true

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  cd $REPO
  export PYTHONPATH=$REPO
  python cluster/check_contract.py --model $MODEL
"
