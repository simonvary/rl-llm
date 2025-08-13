set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 43 \
    --machine_name "beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 41 \
    --machine_name "beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 &

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.999999995 \
    --seed 42 \
    --machine_name "1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 &

CUDA_VISIBLE_DEVICES=3,4,5,6,7,0,1,2 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.999999995 \
    --seed 42 \
    --machine_name "1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 &



wait
echo "All runs finished."



set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --machine_name "3gen-beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \
    --num_generations 3 \
    --gradient_accumulation_steps 3 &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.9999999 \
    --seed 41 \
    --machine_name "3gen-beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \
    --num_generations 3 \
    --gradient_accumulation_steps 3 &

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999995 \
    --seed 41 \
    --machine_name "3gen-beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 &

CUDA_VISIBLE_DEVICES=3,4,5,6,7,0,1,2 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999999 \
    --seed 41 \
    --machine_name "3gen-beta0.001-16steps-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001  &



wait
echo "All runs finished."