CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --gamma 1.0 \
    --seed 43 \
    --machine_name "constantlr-1syncsteps-1epoch-58"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 1 \

