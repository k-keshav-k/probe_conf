#!/usr/bin/env bash
# Environment setup for the probe_conf pilot on AMD Strix Halo (gfx1151).
#
# The native ROCm 6.4 PyTorch wheels do NOT contain gfx1151 kernels; the
# gfx1151-specific index below does. Tested working 2026-09-28:
#   torch 2.10.0+rocm7.13.0a20260513, 15.6 TFLOPS fp16 matmul, 57 GB usable.
#
# Stable alternative (AMD official pip docs):
#   pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
#       "torch[device-gfx1151]==2.12.0+rocm7.14.1" \
#       "torchvision[device-gfx1151]==0.27.0+rocm7.14.1" \
#       "torchaudio==2.11.0+rocm7.14.1"
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch
uv pip install --python .venv/bin/python \
    transformers datasets scikit-learn pandas matplotlib tqdm accelerate \
    safetensors sentencepiece protobuf pyarrow scipy markdown reportlab

echo "Done. Verify with: .venv/bin/python scripts/smoke_test.py"
