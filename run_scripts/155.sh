python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --beta 0.4\
    --machine_name "beta0.4-1epoch-155"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \

python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --beta 0.2\
    --machine_name "beta0.2-1epoch-155"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \


python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --beta 0.1\
    --machine_name "beta0.1-1epoch-155"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \

python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --beta 0.01\
    --machine_name "beta0.01-1epoch-155"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \


