python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --gamma 1.0 \
    --seed 44 \
    --machine_name "constantlr-1epoch-capacityblock2"
    --sync_ref_model \
    --disable_dropout \
    --sync_ref_model_steps 16 \
