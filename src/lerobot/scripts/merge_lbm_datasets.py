#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================

# 存放技能数据集的根目录
DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")

# 目标合并后的数据集名称映射
SPLIT_GROUPS = {
    "train": "lbm-eval-train",
    "iid": "lbm-eval-iid",
    "ood_manipuland": "lbm-eval-ood_manipuland",
    "ood_operator": "lbm-eval-ood_operator",
    "ood_skill": "lbm-eval-ood_skill"
}

# 优先级后缀
RESIZED_SUFFIX = "_resized"

# ==============================================================================
# Merge Logic
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    dataset_root = DEFAULT_DATASET_ROOT
    
    # 1. 扫描所有数据集目录
    if not dataset_root.exists():
        logging.error(f"Dataset root {dataset_root} does not exist.")
        return

    all_dirs = [d.name for d in dataset_root.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
    
    # 2. 按 Split 类型分组
    groups = defaultdict(list)
    for d in all_dirs:
        for suffix_key in SPLIT_GROUPS.keys():
            # 匹配后缀，例如 _train 或 _train_resized
            # 注意匹配顺序，防止 ood_manipuland 匹配到 ood
            if f"_{suffix_key}" in d:
                is_resized = d.endswith(RESIZED_SUFFIX)
                # 提取 core_name 用于去重。例如：
                # lbm-eval-skills-xxx_train -> lbm-eval-skills-xxx_train
                # lbm-eval-skills-xxx_train_resized -> lbm-eval-skills-xxx_train
                core_name = d.replace(RESIZED_SUFFIX, "")
                
                groups[suffix_key].append({
                    "core": core_name,
                    "is_resized": is_resized,
                    "full": d
                })
                break

    # 3. 对每组进行去重（优先选择 resized）
    final_merge_lists = {}
    for split_key, datasets in groups.items():
        core_map = defaultdict(list)
        for item in datasets:
            core_map[item["core"]].append(item)
        
        selected_repos = []
        for core, versions in core_map.items():
            # 按 is_resized 排序，True 排在前面
            best_version = sorted(versions, key=lambda x: x["is_resized"], reverse=True)[0]
            selected_repos.append(best_version["full"])
        
        final_merge_lists[split_key] = sorted(selected_repos)

    # 4. 执行合并操作
    for split_key, repo_list in final_merge_lists.items():
        target_repo = SPLIT_GROUPS[split_key]
        if not repo_list:
            logging.warning(f"No datasets found for group {split_key}. Skipping.")
            continue
        if (dataset_root / target_repo).exists():
            logging.info(f"Target repository {target_repo} already exists. Skipping.")
            continue
            
        logging.info(f"--- Merging group [{split_key}] into [{target_repo}] ---")
        logging.info(f"Source datasets ({len(repo_list)}): {repo_list}")
        
        # 构造 lerobot_edit_dataset 命令
        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset",
            "--repo_id", target_repo,
            "--root", str(dataset_root),
            "--operation.type", "merge",
            "--operation.repo_ids", json.dumps(repo_list)
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        src_path = project_root / "src"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(src_path)
        
        try:
            subprocess.run(cmd, env=env, check=True)
            logging.info(f"Successfully merged {target_repo}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to merge {target_repo}. Error code: {e.returncode}")

    logging.info("All merging tasks completed.")

if __name__ == "__main__":
    main()
