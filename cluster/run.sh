#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — run one command inside the section33 Apptainer container.
#
# Use it on any node where the overlay is not already mounted (for example the
# login node).  Examples:
#
#   bash cluster/run.sh python -m section33.cli setup
#   bash cluster/run.sh python -m section33.cli audit
#   bash cluster/run.sh python -m section33.smoke
#
# Jobs use their own sbatch scripts and do not need this helper.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/scratch/$USER/probe_conf}"
source "$REPO/cluster/env.sh"

if [ "$#" -eq 0 ]; then
  echo "usage: run.sh <command> [args...]" >&2
  exit 2
fi

if [ -d /ext3 ]; then
  # Already inside the container.
  source /ext3/env.sh
  cd "$REPO"
  exec "$@"
fi

NV=""
if [ -e /dev/nvidia0 ]; then NV="--nv"; fi

exec apptainer exec $NV --overlay "$OVERLAY:ro" "$SIF" \
  /bin/bash -c '
    source /ext3/env.sh
    export HF_HOME="'"$HF_HOME"'"
    export TORCH_HOME="'"$TORCH_HOME"'"
    export TRITON_CACHE_DIR="'"$TRITON_CACHE_DIR"'"
    export TMPDIR="'"$TMPDIR"'"
    cd "'"$REPO"'"
    exec "$@"
  ' _ "$@"
