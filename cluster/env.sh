# ---------------------------------------------------------------------------
# NYU Torch settings for the section33 experiment.
# Copy this file, edit the values, and `source cluster/env.sh` before work.
# ---------------------------------------------------------------------------

# Your Slurm account from `my_slurm_accounts` (required for every job).
export TORCH_ACCOUNT="${TORCH_ACCOUNT:-torch_pr_XXX_XXXXX}"

# Scratch space. Code, data, model cache, overlays and run output live here.
# Override REPO before sourcing to keep the code in /home (recommended there:
# it is backed up and never purged).  Jobs use $SLURM_SUBMIT_DIR when the
# repository is submitted with --chdir.
export REPO="${REPO:-/scratch/$USER/probe_conf}"
export RUN_DIR="/scratch/$USER/section33_v1"
export HF_HOME="/scratch/$USER/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="/scratch/$USER/torch_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"

# Container assets from the NYU docs (verify with `ls /share/apps/images`).
export OVERLAY="/scratch/$USER/section33_overlay-15GB-500K.ext3"
export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY_SRC="/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz"

# Model and data choices for this experiment.
export PRIMARY_MODEL="Qwen/Qwen2.5-7B-Instruct"
export REPLICATION_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# Keep caches and temporary files out of $HOME (50 GB / 30 K inodes).
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
export UV_CACHE_DIR="/scratch/$USER/uv_cache"
export TMPDIR="/scratch/$USER/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" \
         "$UV_CACHE_DIR" "$TMPDIR" 2>/dev/null || true

# Threads follow the Slurm allocation inside jobs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
