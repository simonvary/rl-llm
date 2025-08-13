set -euo pipefail
mkdir -p eval_results
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.9999999 \
    --seed 42 \
    --machine_name "beta0.1-31"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.1 &


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.9999999 \
    --seed 42 \
    --machine_name "beta0.01-31"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.01 &


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999995 \
    --seed 42 \
    --machine_name "beta0.1-31"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.1 & 


CUDA_VISIBLE_DEVICES=3,4,5,6,7,0,1,2 python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999995 \
    --seed 42 \
    --machine_name "beta0.01-31"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.01 &

wait
echo "All runs finished."




  