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
  if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: overlay file not found: $OVERLAY" >&2
    echo "Create it inside an interactive compute session:" >&2
    echo "  srun --account=\"\$TORCH_ACCOUNT\" --cpus-per-task=4 --mem=16GB --time=02:00:00 --pty /bin/bash" >&2
    echo "  cp -rp \"$OVERLAY_SRC\" \"$OVERLAY.gz\" && gunzip -f \"$OVERLAY.gz\"" >&2
    echo "Then run cluster/01_setup_overlay.sh (cluster/README.md section 4)." >&2
    exit 1
  fi
  echo "== entering the section33 container =="
  export S33_IN_CONTAINER=1
  exec apptainer exec --overlay "$OVERLAY:ro" "$SIF" \
    /bin/bash "$REPO/cluster/02_prefetch.sh"
fi

if [ ! -f /ext3/env.sh ]; then
  echo "ERROR: /ext3/env.sh is missing. The overlay environment is not built." >&2
  echo "Run the one-time setup on a compute node first (cluster/README.md section 4):" >&2
  echo "  srun --account=\"\$TORCH_ACCOUNT\" --cpus-per-task=4 --mem=16GB --time=02:00:00 --pty /bin/bash" >&2
  echo "  apptainer exec --fakeroot --overlay \"\$OVERLAY:rw\" \"\$SIF\" /bin/bash" >&2
  echo "  bash \$REPO/cluster/01_setup_overlay.sh" >&2
  exit 1
fi
if [ ! -x /ext3/.venv/bin/python ]; then
  echo "ERROR: /ext3 has no Python environment (/ext3/.venv missing)." >&2
  echo "Rerun cluster/01_setup_overlay.sh. It needs internet on the compute node." >&2
  exit 1
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
  python -m section33.data.wikidata --out "$RUN_DIR/source/wikidata_cohort.csv" \
      --per-relation "${WIKIDATA_PER_RELATION:-10000}"
else
  echo "Skipped. Set FETCH_WIKIDATA=1 to fetch a fresh Wikidata cohort."
fi

# Confirmatory mode requires a popularity stratum.  Three options:
#   POPULARITY_SOURCE=sitelinks  (fast proxy: number of Wikipedia editions)
#   FETCH_PAGEVIEWS=1            (monthly views from the Wikimedia API)
#   PAGEVIEW_CSV=subject_id,popularity.csv  (a frozen dump)
if [ "${POPULARITY_SOURCE:-}" = "sitelinks" ] && [ -f "$RUN_DIR/source/wikidata_cohort.csv" ]; then
  echo "== popularity from Wikidata sitelinks (language editions) =="
  python - <<PY
import pandas as pd

path = "$RUN_DIR/source/wikidata_cohort.csv"
df = pd.read_csv(path)
if "sitelinks" not in df.columns or df["sitelinks"].isna().all():
    raise SystemExit("no sitelinks column; refetch with FETCH_WIKIDATA=1 on the current code")
df["popularity"] = df["sitelinks"]
df.to_csv(path, index=False)
print("popularity set from sitelinks:", int(df["popularity"].notna().sum()), "of", len(df))
PY
  FETCH_PAGEVIEWS=0
  PAGEVIEW_CSV=""
fi

if [ "${FETCH_PAGEVIEWS:-0}" = "1" ] && [ -f "$RUN_DIR/source/wikidata_cohort.csv" ]; then
  echo "== fetching monthly pageviews (can take several minutes) =="
  python -m section33.data.pageviews \
    --cohort "$RUN_DIR/source/wikidata_cohort.csv" \
    --out "$RUN_DIR/source/pageviews.csv" \
    --workers "${PAGEVIEW_WORKERS:-8}"
  PAGEVIEW_CSV="$RUN_DIR/source/pageviews.csv"
fi

if [ -n "${PAGEVIEW_CSV:-}" ]; then
  if [ ! -f "$PAGEVIEW_CSV" ]; then
    echo "ERROR: PAGEVIEW_CSV not found: $PAGEVIEW_CSV" >&2
    echo "Supply a real subject_id,popularity CSV or set FETCH_PAGEVIEWS=1." >&2
    exit 1
  fi
  if [ ! -f "$RUN_DIR/source/wikidata_cohort.csv" ]; then
    echo "ERROR: no Wikidata cohort to join; set FETCH_WIKIDATA=1 first." >&2
    exit 1
  fi
  echo "== joining pageviews from $PAGEVIEW_CSV =="
  python - <<PY
import pandas as pd
from pathlib import Path
from section33.data.wikidata import attach_popularity

cohort = pd.read_csv("$RUN_DIR/source/wikidata_cohort.csv")
joined = attach_popularity(cohort, Path("$PAGEVIEW_CSV"))
joined.to_csv("$RUN_DIR/source/wikidata_cohort.csv", index=False)
print(f"popularity joined for {int(joined['popularity'].notna().sum())} of {len(joined)} rows")
PY
fi

echo "Prefetch done."
