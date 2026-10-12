#!/bin/bash
# ---------------------------------------------------------------------------
# Retrieval monitoring S3 — development sweep over layer, rank, alpha (GPU).
#
# Submit from the login node:
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
#          retrieval_monitoring/cluster/06_s3_sweep.sbatch
#
# The first pilot ran at alpha=0.5, moved the subject state by about one
# percent of its norm, and showed no damage.  This job measures the
# teacher-forced reference-answer log-probability on famous subjects for a
# grid of settings, so the pilot can be rerun with settings that have a chance
# of selective damage.  The proxy is development evidence; the pilot still
# measures answer success.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rm-sweep
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=02:00:00
#SBATCH --output=rm-sweep-%j.out
#SBATCH --error=rm-sweep-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

MODEL="${MODEL:-$PRIMARY_MODEL}"
SOURCE="${SOURCE:-$RUN_DIR/source/wikidata_cohort.csv}"
PAIRS="${PAIRS:-$RUN_DIR/retrieval_monitoring/pilot_pairs.csv}"
SUBJECTS="${SUBJECTS:-12}"
LAYERS="${LAYERS:-8 13 18}"
RANKS="${RANKS:-2 8}"
ALPHAS="${ALPHAS:-1 5 10 25}"
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
echo "subjects=$SUBJECTS layers=$LAYERS ranks=$RANKS alphas=$ALPHAS relations=$RELATIONS"

srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -e
  source /ext3/env.sh
  export HF_HOME=$HF_HOME
  export TORCH_HOME=$TORCH_HOME
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  export TOKENIZERS_PARALLELISM=false
  cd $REPO
  python -m retrieval_monitoring.cli sweep \
    --source $SOURCE --pairs $PAIRS --run $RUN_DIR --model $MODEL \
    --subjects $SUBJECTS --layers $LAYERS --ranks $RANKS --alphas $ALPHAS \
    --dtype float16 --relations $RELATIONS
"
