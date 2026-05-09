# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-iid \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/iid_episode_dirs.txt \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-ood-operator \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/ood_operator_episode_dirs.txt \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-ood-manipuland \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/ood_manipuland_episode_dirs.txt \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"

# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-39-train \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/train_episode_dirs.txt \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"

# 12 skills for validation and evaluation
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-12-iid \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/iid_episode_dirs.txt \
#     --skill_types "BimanualPlaceAppleFromBowlIntoBin,BimanualPlaceFruitFromBowlIntoBin,BimanualPutSpatulaOnPlateFromDryingRack,BimanualStoreCerealBoxUnderShelf,PlaceCupByCoaster,PushCoasterToMug,PutBananaOnSaucer,PutKiwiInCenterOfTable,PutMugOnSaucer,PutSpatulaInUtensilCrock,TurnCupUpsideDown,TurnMugRightsideUp" \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-12-ood-operator \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/ood_operator_episode_dirs.txt \
#     --skill_types "BimanualPlaceAppleFromBowlIntoBin,BimanualPlaceFruitFromBowlIntoBin,BimanualPutSpatulaOnPlateFromDryingRack,BimanualStoreCerealBoxUnderShelf,PlaceCupByCoaster,PushCoasterToMug,PutBananaOnSaucer,PutKiwiInCenterOfTable,PutMugOnSaucer,PutSpatulaInUtensilCrock,TurnCupUpsideDown,TurnMugRightsideUp" \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-12-ood-manipuland \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/ood_manipuland_episode_dirs.txt \
#     --skill_types "BimanualPlaceAppleFromBowlIntoBin,BimanualPlaceFruitFromBowlIntoBin,BimanualPutSpatulaOnPlateFromDryingRack,BimanualStoreCerealBoxUnderShelf,PlaceCupByCoaster,PushCoasterToMug,PutBananaOnSaucer,PutKiwiInCenterOfTable,PutMugOnSaucer,PutSpatulaInUtensilCrock,TurnCupUpsideDown,TurnMugRightsideUp" \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"

# # skill ood dataset
# python lerobot/scripts/lbm_dataset_convert.py \
#     --lbm_root /nfs_gaoyang/LBM_dataset \
#     --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
#     --repo_id lbm-eval-5-ood-skill \
#     --fps 10 \
#     --use_videos True \
#     --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/ood_skill_episode_dirs.txt \
#     --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
#     --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"

# visual ood
python lerobot/scripts/lbm_dataset_convert.py \
    --lbm_root /nfs_gaoyang/LBM_augmented \
    --output_root /cephfs/huanghaoxu/Data/LBM_lerobot_dataset \
    --repo_id lbm-eval-12-ood-visual \
    --fps 10 \
    --use_videos True \
    --episode_dir_list /nfs_gaoyang/LBM_dataset/splits/iid_episode_dirs.txt \
    --skill_types "BimanualPlaceAppleFromBowlIntoBin,BimanualPlaceFruitFromBowlIntoBin,BimanualPutSpatulaOnPlateFromDryingRack,BimanualStoreCerealBoxUnderShelf,PlaceCupByCoaster,PushCoasterToMug,PutBananaOnSaucer,PutKiwiInCenterOfTable,PutMugOnSaucer,PutSpatulaInUtensilCrock,TurnCupUpsideDown,TurnMugRightsideUp" \
    --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
    --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
