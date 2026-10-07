#!/usr/bin/env bash
# End-to-end pilot run (after `bash scripts/setup_env.sh`).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL=${PROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-pilot7b}

echo "== 1. data =="
$PY scripts/prepare_data.py --force --n 1200 --mix head

echo "== 2. collect labels + features =="
$PY scripts/collect.py --model "$MODEL" --k 30 --batch-questions 8 \
    --tag "$TAG" --fewshot

echo "== 3. probes (frac label, with raw MLP probes) =="
$PY scripts/probe_eval.py --tag "$TAG" --raw-probes

echo "== 4. probes (greedy label) =="
$PY scripts/probe_eval.py --tag "$TAG" --label greedy --raw-probes

echo "== 5. combined probes and saved out-of-fold scores =="
$PY scripts/extra_probes.py --tag "$TAG" --label frac
$PY scripts/extra_probes.py --tag "$TAG" --label greedy

echo "Done. See results/${TAG}_report.md and results/${TAG}_greedy_report.md"
