# Setup

This guide will walk you through setting up the required environment to run the scripts in this repository.

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:
* [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html)
* [**uv**](https://github.com/astral-sh/uv) (`pip install uv`)

***

### 2. Create and Activate the Conda Environment

First, we'll create a dedicated `conda` environment with a specific Python version. This isolates our dependencies and ensures reproducibility.


Create a conda environment named "verifiers" with Python 3.11
```bash
conda create -n verifiers python=3.11
```
Activate the new environment
```bash
conda activate verifiers
```
Navigate to the project directory (e.g., cd ~/verifiers)
Sync the environment with the project's dependencies
```bash
uv sync --extra all --active
```
Uninstall versions of torch, flash-attn, and vLLM
```bash
uv pip uninstall -y torch torchaudio torchvision flash-attn vllm triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
```
Install the correct CUDA-enabled PyTorch build
```bash
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
Install FlashAttention and vLLM without build isolation to ensure they find the new PyTorch installation
```bash
uv pip install flash-attn --no-build-isolation
uv pip install vllm --no-build-isolation
```
Check the PyTorch version and its CUDA linkage
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}');"
```
Check that FlashAttention is importable
```bash
python -c "import flash_attn; print('Flash Attention imported successfully!')"
```
