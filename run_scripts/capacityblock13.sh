
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 train_gpt.py \
    --seed 42 \
    --machine_name "big-capacityblock13" \
    --sync_ref_model \
    --disable_dropout \
    --shuffle_dataset \
    --gamma 1.0 \
    --use_vllm \
    --attn_implementation flash_attention_2


