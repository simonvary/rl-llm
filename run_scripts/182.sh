python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 0.9999999 \
    --seed 43 \
    --machine_name "constantlr-1epoch-182"
    --sync_ref_model \
    --disable_dropout \
    --sync_ref_model_steps 32 \

