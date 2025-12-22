#!/bin/bash
# echo "Waiting for the last program to exit..."
# pid=61293
# # 等待 PID 对应的进程退出
# while ps -p $pid > /dev/null 2>&1; do
#   sleep 1
# done
echo "Start next program."
python src/lerobot/scripts/lbm_dataset_convert.py \
    --lbm_root /nfs_gaoyang/LBM_dataset \
    --output_root /data/LBM_lerobot_dataset \
    --repo_id lbm-eval \
    --fps 10 \
    --use_videos True \
    --state_keys "robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__grippers__left::panda_hand,robot__actual__grippers__right::panda_hand" \
    --keep_cameras scene_left_0,scene_right_0,wrist_left_plus,wrist_right_minus