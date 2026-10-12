#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pull Torch run outputs back to the laptop.  Run this on your LAPTOP.
#
#   bash cluster/sync_from_torch.sh              # normal pull
#   bash cluster/sync_from_torch.sh --dry-run    # show what would change
#
# Default source is the retrieval-monitoring output directory of the
# section33_v1 run root (sweep, pilot, S1/S2 reports, pairs and rows).  Pull
# the whole run root or a single file by overriding SRC/DEST:
#
#   SRC="kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/" \
#   DEST="runs/section33_v1/" \
#     bash cluster/sync_from_torch.sh
#
# Torch does not support SSH keys.  Each new connection asks for the NetID
# password and Duo.  The ControlMaster lines in cluster/README.md let several
# transfers reuse one authenticated connection for eight hours.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$HERE")}"
SRC="${SRC:-kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/retrieval_monitoring/}"
DEST="${DEST:-$ROOT/runs/section33_v1/retrieval_monitoring/}"

OPTS=(-avz --progress)
if [ "${1:-}" = "--dry-run" ]; then
  OPTS+=(--dry-run)
fi

mkdir -p "$DEST"
echo "source: $SRC"
echo "dest:   $DEST"
rsync "${OPTS[@]}" "$SRC" "$DEST"
