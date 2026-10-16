# causal_confidence cluster runs (NYU Torch)

The package follows the same Apptainer workflow as `section33` and
`selective_metacognition`. Run the stages in order; each writes into
`$CAUSAL_RUN_DIR` (defaults to `$RUN_DIR/causal_confidence/cc_v1/<run_id>`).

```bash
cd /scratch/kk6081/probe_conf
git pull --ff-only
source cluster/env.sh

# P0: reporter positive control on regenerated toy rows
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" causal_confidence/cluster/01_p0.sbatch

# P1: episode contract + discovery answer audit
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" causal_confidence/cluster/02_p1.sbatch

# P2 discovery: freeze one manipulation (read p2_discovery.json before continuing)
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" causal_confidence/cluster/03_p2_discover.sbatch

# P2 validation: one locked evaluation on the independent validation pool
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" causal_confidence/cluster/04_p2_validate.sbatch

# Knowledge-variant feasibility (revised protocol; one seed per job)
MODEL=Qwen/Qwen2.5-7B-Instruct SEED=0 sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
  causal_confidence/cluster/05_knowledge.sbatch
```

Fetch results (mirror ``$RUN_DIR`` on the laptop; replace the run id):

```bash
SRC="kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/causal_confidence/cc_v1/p2/" \
DEST="runs/section33_v1/causal_confidence/cc_v1/p2/" bash cluster/sync_from_torch.sh
```

## What must already exist on the cluster

P1/P2 read the audited question pool from the section 33 run root
(``RUN_DIR`` = ``/scratch/$USER/section33_v1``):

- ``$RUN_DIR/manifests/cohort.parquet``
- ``$RUN_DIR/source/popqa_source.parquet``

Check them before the first P1 job:

```bash
ls -l /scratch/$USER/section33_v1/manifests/cohort.parquet \
      /scratch/$USER/section33_v1/source/popqa_source.parquet
```

If they are missing, push the two files directly (``sync_to_torch.sh``
excludes ``runs/``):

```bash
rsync -avz runs/section33_v1/manifests/cohort.parquet \
  kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/manifests/
rsync -avz runs/section33_v1/source/popqa_source.parquet \
  kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/section33_v1/source/
```

The 7B checkpoint (``Qwen/Qwen2.5-7B-Instruct``) is cached under
``$HF_HOME`` from the section 33 runs; set ``HF_TOKEN`` if the cache is cold.

## Notes

- Inspect `interventions/p2_discovery.json` between jobs 03 and 04: check the
  candidate damage table, the frozen attenuation, and the rescue recovery
  fraction.  The validation job reads the frozen file from the same
  `CAUSAL_RUN_DIR`; nothing else needs to be copied.
- The frozen partition sizes total 2,000 questions plus a 200-question
  transfer set.  The available audited pool (section 33 cohort + PopQA source)
  is smaller, so pass `ALLOW_REDUCED=1` for a pilot and report the achieved
  sizes from `manifests/cohort_split.json`.  A reduced pilot is a pilot, not
  the planned study.
- `p2-validate` refuses to overwrite an existing `p2_validation.json`.  Use a
  new `RUN_ID` for a rerun.
- P3-P6 are gated.  `python -m causal_confidence.cli p3` exits with a message
  until the P2 gates pass.
- The locked gates live in `causal_confidence/protocol.py` (`Gates`).  Freeze
  them before opening validation and do not tune them after.
