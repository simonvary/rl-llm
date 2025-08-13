python train_grpo_discounted.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --machine_name "capacityblock5"\
    --sync_ref_model \
    --disable_dropout \

#aws s3 sync outputs/ s3://starfish-science-dev/ayoualex/outputs_capacityblock4/
