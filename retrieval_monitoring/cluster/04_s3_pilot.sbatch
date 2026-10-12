#!/bin/bash
# ---------------------------------------------------------------------------
# Retrieval monitoring S3 — 40-subject selectivity pilot (GPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          retrieval_monitoring/cluster/04_s3_pilot.sbatch
#
# Builds pilot pairs from the raw Wikidata source (two relations per subject,
# excluding the frozen cohort), fits relation maps on separate training
# subjects, collects 40 subjects x 2 relations x 4 states x 8 draws, and
# reports the selectivity gate.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rm-s3
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=04:00:00
#SBATCH --output=rm-s3-%j.out
#SBATCH --error=rm-s3-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"
SOURCE="${SOURCE:-$RUN_DIR/source/wikidata_cohort.csv}"
PAIRS="${PAIRS:-$RUN_DIR/retrieval_monitoring/pilot_pairs.csv}"
SUBJECTS="${SUBJECTS:-40}"
ALPHA="${ALPHA:-0.5}"
RELATIONS="${RELATIONS:-father occupation}"

if [ ! -f "$PAIRS" ]; then
  echo "ERROR: pairs CSV not found: $PAIRS" >&2
  echo "Fetch it on the login node first (it needs internet):" >&2
  echo "  bash cluster/run.sh python -m retrieval_monitoring.cli fetch-pairs \\" >&2
  echo "      --out \"$PAIRS\" --limit 400" >&2
  exit 1
fi

echo "host: $(hostname)"; nvidia-smi || true
echo "model=$MODEL"
echo "source=$SOURCE"
echo "pairs=$PAIRS"
echo "subjects=$SUBJECTS alpha=$ALPHA relations=$RELATIONS"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export TORCH_HOME=$TORCH_HOME
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  python -m retrieval_monitoring.cli pilot \
    --source $SOURCE --pairs $PAIRS --run $RUN_DIR --model $MODEL \
    --subjects $SUBJECTS --alpha $ALPHA --dtype float16 --relations $RELATIONS
"
