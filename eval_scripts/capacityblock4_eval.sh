#!/usr/bin/env bash
set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # stable GPU indexing

# ---------- capacityblock2 ----------
path='/home/ubuntu/alex/verifiers/outputs/'
date='9-2-'
run=''
name='run2'

# CUDA_VISIBLE_DEVICES=0 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-300" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-1.txt" &

name='Qwen2.5-14B-Instruct-math-gamma1.0-seed42-big-capacityblock9'

CUDA_VISIBLE_DEVICES=1 python huggingface_eval_math.py \
  --model_name "${path}${name}/checkpoint-1000" \
  --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
  > "eval_results/${date}${name}${run}-42-2.txt" &

# #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'
# CUDA_VISIBLE_DEVICES=2 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-900" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-3.txt" &

# # #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'

# CUDA_VISIBLE_DEVICES=3 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-1200" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-4.txt" &


# # #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'

# CUDA_VISIBLE_DEVICES=4 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-1500" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-5.txt" &

# # #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'

# CUDA_VISIBLE_DEVICES=5 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-1714" \
#   --runs 10 --max_model_len 1024 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-6.txt" &

# # #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'


# CUDA_VISIBLE_DEVICES=6 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-1714" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-7.txt" &

# # #name='Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4-10'

# CUDA_VISIBLE_DEVICES=7 python huggingface_eval_math.py \
#   --model_name "${path}${name}/checkpoint-1200" \
#   --runs 10 --max_model_len 3072 --temperature 0.0 --top_p 1.0 --max_tokens 2048 --seed 42 \
#   > "eval_results/${date}${name}${run}-42-8.txt" &



wait
echo "All evals finished."




