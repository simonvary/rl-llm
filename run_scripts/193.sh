python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 43 \
    --machine_name "beta0.4-16steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.4 \

python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 41 \
    --machine_name "beta0.4-16steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.4 \

python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 43 \
    --machine_name "beta0.1-16steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.1 \

python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 41 \
    --machine_name "beta0.1-16steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.1 \

python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 43 \
    --machine_name "beta0.1-32steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \
    --beta 0.1 \

python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 1.0 \
    --seed 41 \
    --machine_name "beta0.1-32steps-1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \
    --beta 0.1 \




