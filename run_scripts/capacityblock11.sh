
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 train_gpt.py \
#     --seed 42 \
#     --machine_name "beta0.01-big-capacityblock11" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --gamma 0.9999 \
#     --use_vllm \
#     --attn_implementation flash_attention_2 \
#     --beta 0.01 \
#     --num_generations 5 \
#     --gradient_accumulation_steps 5

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 train_gpt.py \
#     --seed 42 \
#     --machine_name "beta0.01-big-capacityblock11" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --gamma 0.999 \
#     --use_vllm \
#     --attn_implementation flash_attention_2 \
#     --beta 0.01 \
#     --num_generations 5 \
#     --gradient_accumulation_steps 5


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 train_big.py \
    --seed 42 \
    --machine_name "beta0.01-big-capacityblock11" \
    --sync_ref_model \
    --disable_dropout \
    --shuffle_dataset \
    --gamma 0.9999 \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --beta 0.01 \
    --num_generations 5 \
    --gradient_accumulation_steps 5

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 train_gpt.py \
#     --seed 42 \
#     --machine_name "beta0.01-big-capacityblock11" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --gamma 0.9995 \
#     --use_vllm \
#     --attn_implementation flash_attention_2 \
#     --beta 0.01 \
#     --num_generations 5 \
#     --gradient_accumulation_steps 5

