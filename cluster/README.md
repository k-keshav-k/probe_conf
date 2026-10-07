# Running section33 on NYU Torch

This folder turns the Section 3.3 experiment into NYU Torch (Slurm) jobs.
It follows the official docs at <https://services.rt.nyu.edu/docs/hpc/>
(source: [NYU-RTS/rts-docs](https://github.com/NYU-RTS/rts-docs)). Read that
site once for the full background; this file keeps only what the experiment
needs.

The short version: **log in, read the docs once, get an allocation, put data on
`/scratch`, build one Apptainer overlay, prefetch models on the login node,
then submit GPU jobs. Do not run training or installs on login nodes.**

---

## 1. Access checklist

| Item | Detail |
| --- | --- |
| HPC account | Request at <https://identity.it.nyu.edu/> on the NYU VPN; needs a faculty sponsor |
| Slurm account | Comes from an **active allocation** in <https://projects.hpc.nyu.edu> |
| Account name | Find it with `my_slurm_accounts`; use it as `--account=...` on every job |
| Login | `ssh <NetID>@login.torch.hpc.nyu.edu` (VPN needed off campus) |
| 2FA | Device login at <https://microsoft.com/devicelogin>, then Duo |
| SSH keys | **Not supported** on Torch. Use passwords, or Globus for transfers |
| Web access | <https://ood.torch.hpc.nyu.edu> (Open OnDemand, VPN needed) |
| Support | hpc@nyu.edu |

Edit `cluster/env.sh` once with your `TORCH_ACCOUNT`, paths, and e-mail.

## 2. Storage: where things go

| Path | Quota | Use it for |
| --- | --- | --- |
| `/home/$USER` | 50 GB / 30 K files | Code, git clone, job scripts. Backed up |
| `/scratch/$USER` | 5 TB / 5 M files | Datasets, HF cache, overlays, run output. **No backup** |
| `/archive/$USER` | 2 TB / 20 K files | Long-term archives. Not available to jobs |
| `/projects/work/public/ml-datasets` | read-only | Public datasets (COCO, ImageNet, ...) |

Rules:

- Scratch files **not accessed for 60 days are purged**. Keep code and scripts
  in `/home` or git. Do not `touch` files to defeat the purge.
- Check usage often: `myquota`.
- `$HOME` fills up fast from caches. This repo sends `HF_HOME`, `TORCH_HOME`,
  `TRITON_CACHE_DIR`, pip/uv caches, and `TMPDIR` to `/scratch` (see
  `cluster/env.sh`).
- Large transfers go through the DTN (`dtn.torch.hpc.nyu.edu`) or Globus
  (`nyu#torch`). Do not transfer from login nodes.

## 3. Get the code onto Torch

Push the repo from your laptop, then clone on the login node:

```bash
# laptop
git add . && git commit -m "section33 cluster setup" && git push
# torch login node
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone <your-repo-url> probe_conf
cd probe_conf && git log --oneline -1
```

The pilot label files (`results/*labels*.csv`) are small and needed for cohort
exclusion; make sure they are committed and arrive with the clone. The raw
pilot tensors (`results/raw/`) stay local.

## 4. One-time environment (Apptainer overlay + uv)

```bash
ssh <NetID>@login.torch.hpc.nyu.edu
source /scratch/$USER/probe_conf/cluster/env.sh
ls /share/apps/images | grep -i cuda      # verify $SIF exists
srun --account="$TORCH_ACCOUNT" --cpus-per-task=4 --mem=16GB \
     --time=02:00:00 --pty /bin/bash
# inside the compute node session:
cp -rp "$OVERLAY_SRC" "$OVERLAY.gz" && gunzip -f "$OVERLAY.gz"
apptainer exec --fakeroot --overlay "$OVERLAY:rw" "$SIF" /bin/bash
# inside the container:
bash /scratch/$USER/probe_conf/cluster/01_setup_overlay.sh
exit; exit
```

Why a compute node: login nodes have a 2 GB memory limit and a different OS
image, so package installs can fail or produce binaries that do not run.
Production jobs mount the overlay **read-only** (`:ro`), which lets many jobs
share it safely.

## 5. Prefetch models and data (login node, after step 4)

Compute nodes usually have **no internet**, so download first. The script
enters the container by itself:

```bash
source /scratch/$USER/probe_conf/cluster/env.sh
bash /scratch/$USER/probe_conf/cluster/02_prefetch.sh
```

For a gated model such as Mistral, accept the license on the model page and
export `HF_TOKEN` first. For a fresh Wikidata cohort:

```bash
FETCH_WIKIDATA=1 bash /scratch/$USER/probe_conf/cluster/02_prefetch.sh
```

The fresh cohort still needs pageview popularity and the other metadata before
`04_cohort` can match the plan; see the comment at the top of
`section33/data/wikidata.py`.

## 6. The run, stage by stage

Stage gates follow `report/SECTION_3_3_TEST_PLAN.md` section 13.

```bash
source /scratch/$USER/probe_conf/cluster/env.sh
cd "$REPO"

# 0. freeze the protocol and run layout (inside the container)
bash cluster/run.sh python -m section33.cli setup

# 1. gates: unit tests + small-model smoke on one GPU
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/03_smoke.sbatch
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
       --export=ALL,MODEL="$PRIMARY_MODEL" cluster/07_gpu_contract.sbatch

# 2. cohort and manifests (CPU)
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
       --export=ALL,COHORT_SOURCE="$RUN_DIR/source/wikidata_cohort.csv" \
       cluster/04_cohort.sbatch

# 3. collection, one track and split at a time, sharded
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-3 \
       --export=ALL,TRACK=A,SPLIT=train,SHARDS=4,MODEL="$PRIMARY_MODEL" \
       cluster/05_collect.sbatch
# repeat for dev, calibration, test; then TRACK=B (all splits)

# 4. probe selection, calibration, and the frozen evaluation (CPU)
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" \
       --export=ALL,MODEL="$PRIMARY_MODEL" cluster/06_evaluate.sbatch
```

Useful checks while jobs run:

```bash
squeue --me                          # queue (do not run it under `watch`)
seff <jobid>                         # CPU/memory efficiency of a finished job
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,AllocCPUS
sinfo --Format=Partition,GRES,CPUs,Features:26,NodeList
show_slurm_qos                       # wall-time and resource limits
myquota                              # disk and inode usage
```

## 7. Updating the code on Torch

The cluster copy is a normal repository directory. Update it in one of two ways.

**Option A — rsync (no git remote needed).** From your laptop:

```bash
bash cluster/sync_to_torch.sh              # add --delete to remove deleted files
```

This uses the excludes that keep `.venv`, raw tensors, run data, and caches
local. Only changed files are sent and the transfer is a few MB. On the login
node, verify with:

```bash
cd /scratch/kk6081/probe_conf && git status --short | head
```

**Option B — git (best for repeated updates).** Commit and push locally, then
pull on the login node (login nodes have internet):

```bash
# laptop
cd ~/Research/probe_conf
git add -A && git commit -m "fix: review findings"
git push origin main

# torch login node
cd /scratch/kk6081/probe_conf
git pull --ff-only
```

If the repository has no remote, use a git bundle instead of GitHub:

```bash
# laptop
git bundle create /tmp/probe_conf.bundle --all
scp /tmp/probe_conf.bundle kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/
# login node
cd /scratch/kk6081/probe_conf && git pull /scratch/kk6081/probe_conf.bundle main
```

**Reduce Duo prompts.** Add a connection cache to `~/.ssh/config` on the laptop:

```text
Host dtn.torch.hpc.nyu.edu login.torch.hpc.nyu.edu
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
```

The first transfer authenticates with Duo; later transfers within eight hours
reuse the connection.

**Two rules.**

- New `sbatch` submissions use the new code. A running job keeps the version it
  started with. To apply an update mid-run, `scancel` the job and submit again;
  the collector resumes from the last finished question.
- Keep code under `/home` (backed up, never purged) and data under `/scratch`
  (5 TB, purged after 60 idle days). The scripts accept `REPO=/home/<NetID>/probe_conf`;
  submit with `--chdir="$REPO"`.

## 8. Slurm recipes used here

| Need | Flag |
| --- | --- |
| Account | `--account=torch_pr_XXX_XXXXX` (every job) |
| GPUs | `--gres=gpu:1`, `--constraint="l40s|h200"` |
| Many shards | `#SBATCH --array=0-3`, use `$SLURM_ARRAY_TASK_ID` |
| Wall time | `--time=HH:MM:SS` or `D-HH:MM:SS` |
| Mail | `--mail-type=END,FAIL --mail-user=<NetID>@nyu.edu` |
| Preemptible use of stakeholder GPUs | `--comment="preemption=yes;requeue=true"` |
| Whole GPU per job | `--gres=gpu:N` and `--ntasks-per-node=N` for DDP |

Facts that shape the scripts:

- **GPU quota:** up to 24 GPUs per user for jobs shorter than 48 h.
- **Wall-time classes:** `gpu168` allows 7 days and up to 4 GPUs per job;
  `gpu48` allows 2 days and up to 16 GPUs. Shorter classes start sooner.
- **Low GPU use gets jobs cancelled** (50% for `l40s`, 60% for `h200`;
  warnings at 70-75%). Keep batches large and avoid idle waits.
- **Preempted jobs are requeued** (eligible after 30 minutes) and the
  collector resumes from where it stopped.
- Do not choose partitions manually; the scheduler maps requests to the right
  QoS. Preemption is the only documented exception.

## 9. Best-practice checklist

- [ ] Jobs (not login nodes) for training, feature extraction, and installs.
- [ ] Login nodes only for editing, `sbatch`, short prefetch downloads.
- [ ] Data and caches on `/scratch`; code and scripts on `/home` or git.
- [ ] Request only the GPUs/CPUs/RAM/time the job needs; check with `seff`.
- [ ] Test on 8-50 questions before the full run (`LIMIT=8`).
- [ ] Resumable jobs: successful work is skipped on requeue.
- [ ] Use mail notifications instead of polling `squeue`.
- [ ] Use job arrays for shards to lower scheduler load.
- [ ] Prefetch models on the login node; compute nodes may be offline.
- [ ] Check `myquota` before and after long runs.
- [ ] Archive results to `/archive` or Globus when the study ends.
- [ ] Cite the cluster: "This work was supported in part through the NYU IT
      High Performance Computing resources, services, and staff expertise."

## 10. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `sbatch: Invalid account` | No active allocation or wrong name | `my_slurm_accounts`, then `--account=...` |
| `Disk quota exceeded` under `~/.cache/huggingface` | HF cache in `$HOME` | `export HF_HOME=/scratch/$USER/hf_cache` |
| `No module named transformers` in a job | Environment not activated | `source /ext3/env.sh` inside the container command |
| `overlay in use by another process` | A shell still holds the overlay in `:rw` | Close it, or mount `:ro` for jobs |
| `FATAL: could not create upper dir` | Overlay not made by Apptainer | Use `apptainer overlay create` or the supplied `overlay-*.ext3.gz` |
| Job killed for low GPU use | Idle or tiny batches | Larger batches; remove sleeps; check `nvidia-smi` |
| Job killed at wall time | Time limit too small | Increase `--time`; the collector resumes |
| `SHARDS` greater than 1 needs a matching array | `--array` was not set | Submit with `--array=0-$((SHARDS-1))`; the script exits with a message otherwise |
| `sbatch` job cannot download a model | No internet on compute nodes | Prefetch on the login node |
| Cannot ssh with a key | Keys are disabled on Torch | Use password + Duo, or Globus |

## 11. Files in this folder

| File | Purpose |
| --- | --- |
| `env.sh` | Account, scratch paths, caches, model names. Edit once. |
| `requirements.txt` | Python packages for the overlay environment |
| `run.sh` | Run one command inside the container (login-node helper) |
| `sync_to_torch.sh` | Push code updates from the laptop to the Torch DTN |
| `01_setup_overlay.sh` | Build the overlay venv with uv (run inside the container) |
| `02_prefetch.sh` | Download models and the PopQA table (run on the login node) |
| `03_smoke.sbatch` | Unit tests and small-model end-to-end smoke (1 GPU) |
| `04_cohort.sbatch` | Build the cohort and split manifest (CPU) |
| `05_collect.sbatch` | Sharded collection for one track/split (1 GPU per array task) |
| `06_evaluate.sbatch` | Development selection, calibration, final evaluation (CPU) |
| `07_gpu_contract.sbatch` | Bias/reconstruction/feature gate for the real model (1 GPU) |
| `check_contract.py` | Python helper for `07_gpu_contract.sbatch` |
