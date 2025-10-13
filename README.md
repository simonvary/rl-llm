# ReadME

This repository contains scripts for training language models on mathematical reasoning tasks like GSM8K and MATH, using Generative Rejection-based Policy Optimization (GRPO).

---

## ⚙️ Prerequisites

* **Conda:** You must have Anaconda or Miniconda installed.
* **NVIDIA GPUs:** The scripts are configured by default for a machine with at least **two** NVIDIA GPUs (one for the vLLM server, one for training). Instructions for single-GPU usage are provided below.
* **Python:** The environment will be set up with Python 3.11.

---

## 🚀 Getting Started

Follow these steps to set up your environment and run the training scripts.

### 1. Create and Activate Conda Environment

First, you need to create a new conda environment with Python 3.11 and activate it.

```bash
conda create -n grpo_env python=3.11
conda activate grpo_env
```

### 2. Run the Setup Script

Once the environment is active, run the provided setup script to install all the necessary dependencies.

```bash
bash setup.sh
```

---

## ▶️ How to Run

Running the training process involves two main steps: starting the vLLM inference server and then launching the training script.

### 1. Start the vLLM Server

In a new terminal, run the following command to start the vLLM server. This will host the `Qwen/Qwen2.5-7B-Instruct` model on **GPU 0**.

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --data-parallel-size 1
```

### 2. Run the Training Script

In a separate terminal (with the `grpo_env` still active), run the training script. This example uses `train_gsm8k.py` and assigns the training process to the remaining GPUS.

```bash
CUDA_VISIBLE_DEVICES=1,2,... python train_gsm8k.py
```

You can replace `train_gsm8k.py` with `train_XXX.py` to run the other training script (different dataset/models).

---


### Single GPU Usage

If you have only **one GPU**, you cannot run the vLLM server and the training script simultaneously. You must disable the vLLM-based generation in the script.

1.  Open either `train_gsm8k.py` or `train_math.py`.
2.  Locate the `GRPOConfig` arguments.
3.  Set `use_vllm = False`.
4.  Comment out the `generation_kwargs` line directly below it.

## 📬 Contact

For any questions or suggestions, please feel free to reach out at [aayoub@ualberta.ca](mailto:aayoub@ualberta.ca).
