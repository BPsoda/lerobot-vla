# FLASH_ATTENTION_DISABLE=1 python -m lerobot.scripts.lbm_eval_policy_server \
#     --model-type pi05 \
#     --checkpoint-path outputs/train/lbm-eval-train-pi05/checkpoints/006000/pretrained_model \
#     --server-uri localhost:50051 \
#     --no-compile-model \
#     --task-instruction "pick and place box" \
#     --n-action-step 4   # Set 1 or 4 for better performance
python -m lerobot.scripts.lbm_eval_policy_server \
    --model-type smolvla \
    --checkpoint-path outputs/train/lbm-eval-train-smolvla-2025-12-19-15-25-06/checkpoints/020000/pretrained_model \
    --server-uri localhost:50051 \
    --no-compile-model \
    --task-instruction "pick and place box" \
    --n-action-step 4   # Set 1 or 4 for better performance