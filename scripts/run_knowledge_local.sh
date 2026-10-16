#!/bin/bash
# ---------------------------------------------------------------------------
# Local asynchronous knowledge-variant feasibility runs.
#
# Each seed is one independent experiment (variants, verification, reporter
# training, transfer) writing to its own run directory.  Runs start in the
# background; follow them with `tail -f <out_root>/seed*.log`.
#
# Usage:
#   bash scripts/run_knowledge_local.sh                 # seeds 0 1 2
#   SEEDS="0" bash scripts/run_knowledge_local.sh       # one seed
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct SEEDS="0 1" \
#       bash scripts/run_knowledge_local.sh
#
# Notes:
# - Quven2.5-0.5B is the default; larger models fit in 57 GB but run slower.
# - Re-running with the same OUT_ROOT reuses trained variant checkpoints
#   (delete <out_root>/seed<N>/variants to retrain from scratch).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS="${SEEDS:-0 1 2}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_ROOT="${OUT_ROOT:-runs/causal_confidence/cc_v1/knowledge}"
EPOCHS="${EPOCHS:-40}"
REPORTER_STEPS="${REPORTER_STEPS:-1200}"
DEVICE="${DEVICE:-}"
EXTRA="${EXTRA:-}"
# Concurrent seeds.  Keep this at 1 on the local AMD GPU: several training
# processes together triggered a ROCm-level teardown and killed the batch.
PARALLEL="${PARALLEL:-1}"

mkdir -p "$OUT_ROOT"
echo "model: $MODEL"
echo "seeds: $SEEDS (parallel=$PARALLEL)"
echo "out:   $OUT_ROOT"

for seed in $SEEDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
  log="$OUT_ROOT/seed${seed}.log"
  echo "launching seed $seed -> $log"
  .venv/bin/python -u -m causal_confidence.cli knowledge \
    --seed "$seed" \
    --model "$MODEL" \
    --out "$OUT_ROOT/seed${seed}" \
    --epochs "$EPOCHS" \
    --reporter-steps "$REPORTER_STEPS" \
    ${DEVICE:+--device "$DEVICE"} \
    $EXTRA > "$log" 2>&1 &
done
wait
echo "all seeds finished"
