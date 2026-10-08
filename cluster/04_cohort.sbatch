#!/bin/bash
# ---------------------------------------------------------------------------
# 04 — Build the frozen cohort and split manifest (CPU job).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          --export=ALL,COHORT_SOURCE="$RUN_DIR/source/wikidata_cohort.csv" \
#          cluster/04_cohort.sbatch
#
# The source table must have these columns (aliases accepted):
#   question_id, subject_id, subject_name, relation, question,
#   reference_answers, popularity, source_revision
#
# Build the fresh cohort before collection (plan section 4).  For a quick
# pipeline test on Torch you may use the local PopQA export with smaller
# splits instead:
#   ... --export=ALL,COHORT_SOURCE="$RUN_DIR/source/popqa_source.parquet" \
#          DEV_TEST=1 cluster/04_cohort.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=s33-cohort
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=s33-cohort-%j.out
#SBATCH --error=s33-cohort-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
# Resolve the repository: explicit REPO, the submit directory, or the cwd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

COHORT_SOURCE="${COHORT_SOURCE:-$RUN_DIR/source/wikidata_cohort.csv}"
if [ ! -f "$COHORT_SOURCE" ]; then
  echo "ERROR: source table not found: $COHORT_SOURCE" >&2
  echo "Run cluster/02_prefetch.sh with FETCH_WIKIDATA=1, or set COHORT_SOURCE." >&2
  exit 1
fi

# Historical label files keep the fresh cohort disjoint (plan section 4.1).
SEEN=$(ls results/*labels*.csv 2>/dev/null | tr '\n' ' ')
if [ -z "$SEEN" ]; then
  echo "ERROR: no results/*labels*.csv found in the repo. Copy your pilot label" >&2
  echo "files to $REPO/results/ so the cohort can exclude seen subjects." >&2
  exit 1
fi

if [ "${DEV_TEST:-0}" = "1" ]; then
  echo "== development test cohort from the cached PopQA snapshot =="
  # shellcheck disable=SC2086
  srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
    source /ext3/env.sh
    cd $REPO
    python -m section33.cli cohort --run $RUN_DIR --source $COHORT_SOURCE \
        --seen $SEEN --per-relation 200 --train 300 --dev 150 --calibration 100 --test 183
  "
else
  echo "== fresh cohort (plan section 4) =="
  if [ ! -f "$RUN_DIR/source/wikidata_cohort.csv" ]; then
    echo "ERROR: fresh Wikidata cohort missing. The 733-question cached snapshot is" >&2
    echo "the legacy check only (plan section 4.1), not the powered fresh cohort." >&2
    exit 1
  fi
  # shellcheck disable=SC2086
  srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
    source /ext3/env.sh
    cd $REPO
    python -m section33.cli cohort --run $RUN_DIR \
        --source $RUN_DIR/source/wikidata_cohort.csv \
        --seen $SEEN --per-relation 1500
  "
fi

echo "== manifest =="
cat "$RUN_DIR/manifests/cohort_manifest.json"
