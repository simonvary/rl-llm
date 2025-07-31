# python grpo_discounted.py \
#     --model_name "google/gemma-2b-it" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "constantlr-1syncsteps-1epoch-58"\
#     --sync_ref_model \
#     --disable_dropout \
#     --ref_model_sync_steps 1 \
#     --beta 0.01 

python gemma_grpo.py \
    --model_name "google/gemma-2b-it" \
    --gamma 1.0 \
    --seed 42 \
    --machine_name "constantlr-1syncsteps-1epoch-58"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 1 \
    --beta 0.01 