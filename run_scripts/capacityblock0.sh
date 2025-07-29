python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --gamma 0.99999975 \
    --seed 44 \
    --machine_name "constantlr-1epoch-capacityblock0" \
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
