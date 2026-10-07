#!/bin/bash
# ---------------------------------------------------------------------------
# 01 — Build the Apptainer overlay environment for section33 (run ONCE).
#
# Run this INSIDE an interactive compute node session, inside the container:
#
#   ssh <NetID>@login.torch.hpc.nyu.edu
#   source /scratch/$USER/probe_conf/cluster/env.sh
#   srun --account="$TORCH_ACCOUNT" --cpus-per-task=4 --mem=16GB \
#        --time=02:00:00 --pty /bin/bash
#   cp -rp "$OVERLAY_SRC" "$OVERLAY.gz" && gunzip -f "$OVERLAY.gz"
#   apptainer exec --fakeroot --overlay "$OVERLAY:rw" "$SIF" /bin/bash
#   source /ext3/env.sh   # after the first run creates it
#   bash <repo>/cluster/01_setup_overlay.sh
#
# Compute nodes install packages.  The login nodes have a 2 GB memory limit
# and a different OS image, so never compile there.  Run jobs, not this script,
# on the login node.
# ---------------------------------------------------------------------------
set -euo pipefail

if [ ! -d /ext3 ]; then
  echo "ERROR: /ext3 is not mounted. Start the container first:" >&2
  echo "  apptainer exec --fakeroot --overlay \$OVERLAY:rw \$SIF /bin/bash" >&2
  exit 1
fi

REPO="${REPO:-/scratch/$USER/probe_conf}"
echo "== checking internet from this node =="
if curl -sI --max-time 15 https://astral.sh >/dev/null 2>&1; then
  echo "internet: yes"
  ONLINE=1
else
  echo "internet: NO. Compute nodes often have no internet." >&2
  echo "Fallback: install from the login node, where internet is available." >&2
  echo "See cluster/README.md section 4. If the login node also fails, contact" >&2
  echo "hpc@nyu.edu." >&2
  ONLINE=0
fi

if [ "$ONLINE" = "1" ]; then
  echo "== installing uv into the overlay =="
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/ext3/.uv" sh
  source /ext3/.uv/env

  echo "== creating the project virtual environment =="
  export UV_PYTHON_INSTALL_DIR="/ext3/.uv/python"
  export UV_PROJECT_ENVIRONMENT="/ext3/.venv"
  export UV_CACHE_DIR="/ext3/.uv/cache"
  uv venv /ext3/.venv --python 3.12
  uv pip install --python /ext3/.venv/bin/python -r "$REPO/cluster/requirements.txt"
fi

echo "== writing /ext3/env.sh =="
cat > /ext3/env.sh <<'EOF'
#!/bin/bash
unset -f which
source /ext3/.uv/env
export UV_PYTHON_INSTALL_DIR="/ext3/.uv/python"
export UV_PROJECT_ENVIRONMENT="/ext3/.venv"
export UV_CACHE_DIR="/ext3/.uv/cache"
source /ext3/.venv/bin/activate
export PATH=/ext3/.venv/bin:$PATH
export PYTHONPATH=/ext3/.venv/bin:$PATH
EOF
chmod +x /ext3/env.sh

echo "== verifying the environment =="
source /ext3/env.sh
python -c "import torch, transformers, datasets, sklearn; print('torch', torch.__version__); \
print('cuda', torch.cuda.is_available()); \
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
cd "$REPO" && python -c "import section33; print('section33 import ok', section33.__version__)"

echo
echo "Next:  sbatch cluster/03_smoke.sbatch"
