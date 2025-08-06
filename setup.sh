#!/usr/bin/env bash
set -euo pipefail

# -------- configuration --------
ENV_NAME="alex"        # name of the conda env to create/use
PY_VERSION="3.11"      # Python version to install in that env
# --------------------------------

# 1) Create the env if it doesn't exist yet
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[*] Creating conda env '${ENV_NAME}' (Python ${PY_VERSION}) ..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

# 2) Activate it (load conda helpers first in case this runs inside a script)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[*] Activated conda env:  ${CONDA_PREFIX}"

# 3) Make sure uv is available inside the env
pip install uv

# 4) Tell uv to use *this* env instead of creating .venv
#    –Either export UV_PROJECT_ENVIRONMENT or rely on --active.
export UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}"

echo "[*] Running 'uv sync --extra all' in the active conda env ..."
cd /data2/alex/verifiers
uv sync --extra all --active    # --active forces uv to respect the active env :contentReference[oaicite:0]{index=0}

uv pip uninstall torch torchaudio torchvision flash-attn
uv pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 triton

uv pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA version: {torch.version.cuda}')"


# Install flash-attn in the same env (no build isolation so it can see torch, CUDA, etc.)
pip install flash-attn --no-build-isolation

# Test the installation
python -c "import flash_attn; print('Flash Attention imported successfully')"

uv pip uninstall vllm

uv pip install vllm --no-build-isolation
