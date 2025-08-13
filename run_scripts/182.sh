python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.9999995 \
    --seed 42 \
    --machine_name "1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \


python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.9999999 \
    --seed 42 \
    --machine_name "1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \


python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999995 \
    --seed 42 \
    --machine_name "1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \


python train_grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --gamma 0.99999999 \
    --seed 42 \
    --machine_name "1epoch-182"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --beta 0.001 \






  