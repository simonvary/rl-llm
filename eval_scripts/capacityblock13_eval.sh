#!/usr/bin/env bash
set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # stable GPU indexing

# ---------- capacityblock2 ----------
path='/home/ubuntu/alex/verifiers/outputs/'
date='9-5-'
run=''

name='phi-4-math-gamma0.99925-seed42-beta0.01-big-capacityblock13'

CUDA_VISIBLE_DEVICES=0 python huggingface_eval_aime2025.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-aime25.txt" &


CUDA_VISIBLE_DEVICES=1 python huggingface_eval_olympiad.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-olympiad.txt" &

CUDA_VISIBLE_DEVICES=2 python huggingface_eval_minerva.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-minerva.txt" &

CUDA_VISIBLE_DEVICES=3 python huggingface_eval_amc.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-amc.txt" &


CUDA_VISIBLE_DEVICES=4 python huggingface_eval_aime2025.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-aime25-1.txt" &


CUDA_VISIBLE_DEVICES=5 python huggingface_eval_olympiad.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-olympiad-1.txt" &

CUDA_VISIBLE_DEVICES=6 python huggingface_eval_minerva.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-minerva-1.txt" &

CUDA_VISIBLE_DEVICES=7 python huggingface_eval_amc.py \
  --model_name "${path}${name}/checkpoint-1166" \
  --runs 5 --max_model_len 4608 --temperature 0.0 --top_p 1.0 --max_tokens 3072 --seed 42 \
  > "eval_results/${date}${name}${run}-42-amc-1.txt" &

wait
echo "All evals finished."




