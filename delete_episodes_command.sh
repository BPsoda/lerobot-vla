#!/bin/bash
# 生成删除5000及以后episode的命令

# 首先查看总episode数
python3 << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lbm-eval', root='/nfs_gaoyang/LBM_lerobot_dataset/lbm-eval')
total_episodes = ds.meta.total_episodes
print(f"Total episodes: {total_episodes}")

# 生成要删除的episode索引列表（5000到total_episodes-1）
if total_episodes > 5000:
    indices_to_delete = list(range(5000, total_episodes))
    indices_str = str(indices_to_delete).replace(' ', '')
    print(f"\n要删除的episode索引: {len(indices_to_delete)} 个 (从5000到{total_episodes-1})")
    print(f"\n执行命令:")
    print(f"python -m lerobot.scripts.lerobot_edit_dataset \\")
    print(f"    --repo_id lbm-eval \\")
    print(f"    --root /nfs_gaoyang/LBM_lerobot_dataset/lbm-eval \\")
    print(f"    --operation.type delete_episodes \\")
    print(f"    --operation.episode_indices '{indices_str}'")
else:
    print(f"\n数据集只有 {total_episodes} 个episode，无需删除（索引从0开始，5000及以后不存在）")
EOF

