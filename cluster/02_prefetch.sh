#!/bin/bash
# ---------------------------------------------------------------------------
# 02 — Prefetch models and the PopQA source table.
#
# Run this on a LOGIN node after the overlay environment exists (step 01):
#
#   source /scratch/kk6081/probe_conf/cluster/env.sh
#   bash /scratch/kk6081/probe_conf/cluster/02_prefetch.sh
#
# The script runs inside the container automatically, because the login node
# does not have the Python environment.  Login nodes have internet; compute
# nodes usually do not, so download before submitting jobs.
#
# Notes:
#   - Gated models (e.g. mistralai/Mistral-7B-Instruct-v0.3) need a token.
#     Accept the license on the model page and export HF_TOKEN.
#   - Everything goes to $HF_HOME on /scratch, never to $HOME (50 GB quota).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/scratch/$USER/probe_conf}"
source "$REPO/cluster/env.sh"

# Re-exec inside the container when the overlay is not mounted.
if [ ! -d /ext3 ] && [ "${S33_IN_CONTAINER:-0}" != "1" ]; then
  echo "== entering the section33 container =="
  export S33_IN_CONTAINER=1
  exec apptainer exec --overlay "$OVERLAY:ro" "$SIF" \
    /bin/bash "$REPO/cluster/02_prefetch.sh"
fi

source /ext3/env.sh

echo "== Hugging Face token check =="
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set. Qwen and PopQA work without it."
  echo "Mistral may need it. Export it before this script if needed."
fi

download() {
  local model="$1"
  echo "== downloading $model =="
  if command -v hf >/dev/null 2>&1; then
    hf download "$model" >/dev/null
  else
    python - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("downloaded to", snapshot_download(sys.argv[1]))
PY
  fi
}

download "$PRIMARY_MODEL"
download "Qwen/Qwen2.5-0.5B-Instruct"
download "$REPLICATION_MODEL" || echo "WARNING: replication model not downloaded"

echo "== exporting the PopQA source table =="
mkdir -p "$RUN_DIR/source"
python - <<PY
from datasets import load_dataset
out = "$RUN_DIR/source/popqa_source.parquet"
ds = load_dataset("akariasai/PopQA", split="test")
ds.to_pandas().to_parquet(out, index=False)
print(f"wrote {out}: {len(ds)} questions")
PY

echo "== optional: fresh Wikidata cohort (needs network) =="
if [ "${FETCH_WIKIDATA:-0}" = "1" ]; then
  python -m section33.data.wikidata --out "$RUN_DIR/source/wikidata_cohort.csv"
else
  echo "Skipped. Set FETCH_WIKIDATA=1 to fetch a fresh Wikidata cohort."
fi

echo "Prefetch done."
