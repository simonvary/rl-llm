python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 0.999999995 \
    --seed 43 \
    --machine_name "1epoch-capacityblock3" \
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 8 \
    --warmup_ratio 0.1 \
    --beta 0.1 \

python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 0.999999999 \
    --seed 43 \
    --machine_name "1epoch-capacityblock3" \
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 8 \
    --warmup_ratio 0.1 \
    --beta 0.1 \


python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 0.9999999995 \
    --seed 43 \
    --machine_name "1epoch-capacityblock3" \
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 8 \
    --warmup_ratio 0.1 \
    --beta 0.1 \

aws s3 sync outputs/ s3://starfish-science-dev/ayoualex/outputs_capacityblock3/
