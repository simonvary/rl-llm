python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --machine_name "consbeta0.001tantlr-1epoch-193"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 32 \
    --beta 0.001 \

