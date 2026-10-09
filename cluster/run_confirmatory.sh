#!/bin/bash
# ---------------------------------------------------------------------------
# run_confirmatory.sh — submit the whole confirmatory pipeline as one Slurm DAG.
#
# Run on the Torch login node:
#   cd /scratch/kk6081/probe_conf
#   source cluster/env.sh
#   bash cluster/run_confirmatory.sh
#
# Stages, chained with afterok dependencies (a failure cancels the rest):
#   setup -> prefetch -> cohort -> smoke/contract gates -> collect (one job,
#   every track and split) -> develop+evaluate+audit -> verdict
#
# Collection is a single resumable job with no sharding: one writer per file.
# If it is preempted or hits the wall time, resubmit and it resumes.
#
# Env knobs (defaults in brackets):
#   TEST_SIZE [4000]               final test subjects (must match the protocol)
#   PER_RELATION [1900]            source pool per relation; default is
#                                  (train + dev + calibration + test) / 4
#   TRAIN [2000] DEV [800] CAL [800]
#   BATCH [4]                      questions per collection batch
#   FETCH [1]                      1 = refetch the Wikidata source and models
#   POPULARITY_SOURCE [sitelinks]  sitelinks | pageviews | none
#   REPLICATION [0]                1 = run the replication model after the primary
#   MODEL / REPLICATION_MODEL      protocol defaults
#
# Example with replication:  REPLICATION=1 bash cluster/run_confirmatory.sh
# ---------------------------------------------------------------------------
set -euo pipefail

if [ "${DRY_RUN:-0}" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: run this on a Slurm login node (sbatch not found)." >&2
  exit 1
fi

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO/cluster/env.sh"
cd "$REPO"

TRAIN="${TRAIN:-2000}"
DEV="${DEV:-800}"
CAL="${CAL:-800}"
TEST_SIZE="${TEST_SIZE:-4000}"
PER_RELATION="${PER_RELATION:-$(( (TRAIN + DEV + CAL + TEST_SIZE) / 4 ))}"
BATCH="${BATCH:-4}"
FETCH="${FETCH:-1}"
POPULARITY_SOURCE="${POPULARITY_SOURCE:-sitelinks}"
REPLICATION="${REPLICATION:-0}"
MODEL="${MODEL:-$PRIMARY_MODEL}"
REPLICATION_MODEL="${REPLICATION_MODEL:-$REPLICATION_MODEL}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" = "1" ]; then
  sbatch() {
    echo "DRY_RUN sbatch $*" >&2
    echo "0"
  }
fi

echo "== section33 confirmatory pipeline =="
echo "run root:     $RUN_DIR"
echo "sizes:        train=$TRAIN dev=$DEV calibration=$CAL test=$TEST_SIZE"
echo "per relation: $PER_RELATION (pool $((4 * PER_RELATION)))"
echo "batch:        $BATCH"
echo "model:        $MODEL"
echo "replication:  $REPLICATION"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "== 1-2/6 DRY_RUN: skipping setup and prefetch =="
else
  echo "== 1/6 freeze protocol =="
  bash cluster/run.sh python -m section33.cli setup --run "$RUN_DIR"

  echo "== 2/6 prefetch source and models =="
  if [ "$FETCH" = "1" ]; then
    case "$POPULARITY_SOURCE" in
      sitelinks) FETCH_WIKIDATA=1 POPULARITY_SOURCE=sitelinks bash cluster/02_prefetch.sh ;;
      pageviews) FETCH_WIKIDATA=1 FETCH_PAGEVIEWS=1 bash cluster/02_prefetch.sh ;;
      none)      FETCH_WIKIDATA=1 bash cluster/02_prefetch.sh ;;
      *) echo "ERROR: unknown POPULARITY_SOURCE: $POPULARITY_SOURCE" >&2; exit 1 ;;
    esac
  else
    if [ ! -f "$RUN_DIR/source/wikidata_cohort.csv" ]; then
      echo "ERROR: no source at $RUN_DIR/source/wikidata_cohort.csv; set FETCH=1." >&2
      exit 1
    fi
    echo "using existing source (FETCH=0)"
  fi
fi

echo "== 3/6 submit jobs =="
COHORT=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  --export=ALL,MODE=confirmatory,PER_RELATION="$PER_RELATION",TEST_SIZE="$TEST_SIZE" \
  cluster/04_cohort.sbatch)
GATE_SMOKE=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  cluster/03_smoke.sbatch)
GATE_CONTRACT=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  --export=ALL,MODEL="$MODEL" cluster/07_gpu_contract.sbatch)
GATES="$GATE_SMOKE:$GATE_CONTRACT"

COLLECT=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  --dependency="afterok:$COHORT:$GATES" --export=ALL,MODEL="$MODEL",BATCH="$BATCH" \
  cluster/05_collect_all.sbatch)
EVALUATE=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  --dependency="afterok:$COLLECT" --export=ALL,MODEL="$MODEL" \
  cluster/06_evaluate.sbatch)
VERDICT=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  --dependency="afterok:$EVALUATE" --export=ALL,MODEL="$MODEL" \
  cluster/08_verdict.sbatch)

echo
echo "cohort:     $COHORT"
echo "gates:      $GATE_SMOKE (smoke)  $GATE_CONTRACT (contract)"
echo "collect:    $COLLECT (one job, all tracks and splits)"
echo "evaluate:   $EVALUATE"
echo "verdict:    $VERDICT"

if [ "$REPLICATION" = "1" ]; then
  COLLECT_R=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
    --dependency="afterok:$VERDICT" --export=ALL,MODEL="$REPLICATION_MODEL",BATCH="$BATCH" \
    cluster/05_collect_all.sbatch)
  EVALUATE_R=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
    --dependency="afterok:$COLLECT_R" --export=ALL,MODEL="$REPLICATION_MODEL" \
    cluster/06_evaluate.sbatch)
  VERDICT_R=$(sbatch --parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" \
    --dependency="afterok:$EVALUATE_R" --export=ALL,MODEL="$REPLICATION_MODEL" \
    cluster/08_verdict.sbatch)
  echo "replication collect:  $COLLECT_R"
  echo "replication evaluate: $EVALUATE_R"
  echo "replication verdict:  $VERDICT_R"
fi

mkdir -p "$RUN_DIR/report" 2>/dev/null || true
{
  echo "submitted $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "test_size=$TEST_SIZE per_relation=$PER_RELATION batch=$BATCH"
  echo "model=$MODEL replication=$REPLICATION"
  echo "cohort=$COHORT gates=$GATES"
  echo "collect=$COLLECT"
  echo "evaluate=$EVALUATE verdict=$VERDICT"
} > "$RUN_DIR/report/pipeline_jobs.txt" 2>/dev/null || true

echo
echo "watch:            squeue --me"
echo "collect output:   s33-collect-all-$COLLECT.out"
echo "primary verdict:  s33-verdict-$VERDICT.out"
echo "job record:       $RUN_DIR/report/pipeline_jobs.txt"
