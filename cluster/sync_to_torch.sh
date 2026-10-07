#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push local changes to the Torch Data Transfer Node.  Run this on your LAPTOP.
#
#   bash cluster/sync_to_torch.sh              # normal update
#   bash cluster/sync_to_torch.sh --delete     # also remove files deleted locally
#   DRY_RUN=1 bash cluster/sync_to_torch.sh    # show what would change
#
# The default destination matches the kk6081 layout.  Override it with:
#   TORCH_DEST="<NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/probe_conf/" \
#     bash cluster/sync_to_torch.sh
#
# Torch does not support SSH keys.  Each new connection asks for the NetID
# password and Duo.  Add the ControlMaster lines from cluster/README.md to
# ~/.ssh/config to reuse one authenticated connection for several transfers.
#
# Large and generated files stay local.  The transfer is a few MB and only
# changed files are sent.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(dirname "$HERE")}"
DEST="${TORCH_DEST:-kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/probe_conf/}"

OPTS=(
  -avz --progress
  --exclude '.venv/'
  --exclude 'results/raw/'
  --exclude 'results/*_partial.jsonl'
  --exclude 'runs/'
  --exclude 'data/'
  --exclude '__pycache__/'
  --exclude '.pytest_cache/'
  --exclude '*.log'
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  OPTS+=(--dry-run)
fi
if [ "${1:-}" = "--delete" ]; then
  OPTS+=(--delete)
fi

echo "source: $SRC"
echo "dest:   $DEST"
rsync "${OPTS[@]}" "$SRC/" "$DEST"

cat <<'EOF'

Transfer finished.  On the Torch login node, check the update with:

    cd /scratch/kk6081/probe_conf
    git status --short | head          # if the .git directory was copied
    grep -n "guardrail" section33/analysis/evaluation.py | head

New sbatch submissions use the new code.  Running jobs keep the code they
started with; let them finish or requeue them with `scancel <jobid>` and submit
again (the collector resumes completed work).
EOF
