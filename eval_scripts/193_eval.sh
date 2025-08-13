#!/usr/bin/env bash
set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # stable GPU indexing

# ---------- capacityblock2 ----------
path='/data2/alex/verifiers/outputs/'
date='8-13-'
run=''
name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed43-beta0.001-16steps-1epoch-193'

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.9999999-seed42-1epoch-182'
CUDA_VISIBLE_DEVICES=1 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.9999999-seed41-3gen-beta0.001-16steps-1epoch-193'
CUDA_VISIBLE_DEVICES=2 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-3gen-beta0.001-16steps-1epoch-193'
CUDA_VISIBLE_DEVICES=3 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.99999995-seed41-3gen-beta0.001-16steps-1epoch-193'
CUDA_VISIBLE_DEVICES=4 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.99999999-seed41-3gen-beta0.001-16steps-1epoch-193'
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.99999995-seed42-1epoch-182'
CUDA_VISIBLE_DEVICES=6 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${date}${name}${run}.txt" &

# name='Qwen2.5-0.5B-Instruct-gsm8k-gamma0.999999995-seed42-1epoch-193'
# CUDA_VISIBLE_DEVICES=7 python eval.py \
#   --model_name "${path}${name}/checkpoint-935" \
#   --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
#   > "eval_results/${date}${name}${run}-1.txt" &

# GPU 7 is free; use it if you add another run:
# CUDA_VISIBLE_DEVICES=7 python eval.py ...

wait
echo "All evals finished."
