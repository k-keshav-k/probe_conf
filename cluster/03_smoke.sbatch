#!/bin/bash
# ---------------------------------------------------------------------------
# 03 — GPU smoke test on Torch.
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" \
#          --chdir="/scratch/$USER/probe_conf" \
#          cluster/03_smoke.sbatch
#
# Runs the contract tests and a two-question end-to-end collection with the
# small model.  Keep this job short; use it as a gate before long collection.
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-smoke
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=00:40:00
#SBATCH --output=s33-smoke-%j.out
#SBATCH --error=s33-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
# Resolve the repository: explicit REPO, the submit directory, or the cwd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

echo "host: $(hostname)"; nvidia-smi || true
module purge
srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cd $REPO
  python -m pytest tests/test_section33_contracts.py -q
  python -m section33.smoke --end-to-end --model Qwen/Qwen2.5-0.5B-Instruct
"
