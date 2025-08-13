#!/usr/bin/env bash
set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # stable GPU indexing

# ---------- capacityblock2 ----------
path='/data2/alex/verifiers/outputs_s3/outputs_capacityblock2/'
run='1'

name='Qwen2.5-7B-Instruct-gsm8k-gamma1.0-seed38-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

name='Qwen2.5-7B-Instruct-gsm8k-gamma1.0-seed39-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=1 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

name='Qwen2.5-7B-Instruct-gsm8k-gamma1.0-seed40-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=2 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

name='Qwen2.5-7B-Instruct-gsm8k-gamma1.0-seed41-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=3 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

name='Qwen2.5-7B-Instruct-gsm8k-gamma1.0-seed44-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=4 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

# ---------- base (different path root) ----------
path='/data2/alex/verifiers/outputs_s3/outputs_capacityblock2/Qwen/'

name='Qwen2.5-7B-Instruct-gsm8k-base-seed42-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

name='Qwen2.5-7B-Instruct-gsm8k-base-seed43-constantlr-1epoch-capacityblock2'
CUDA_VISIBLE_DEVICES=6 python eval.py \
  --model_name "${path}${name}/checkpoint-935" \
  --runs 30 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --top_k 0 --max_tokens 786 \
  > "eval_results/${name}${run}.txt" &

# GPU 7 is free; use it if you add another run:
# CUDA_VISIBLE_DEVICES=7 python eval.py ...

wait
echo "All evals finished."
