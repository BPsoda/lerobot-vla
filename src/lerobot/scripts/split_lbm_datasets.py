#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool
import subprocess
import sys

# ==============================================================================
# Configuration
# ==============================================================================

# Directory containing the split files (one directory per skill)
SPLIT_FILE_ROOT = Path("/nfs_gaoyang/LBM_dataset/splits/tasks")

# Directory containing the converted LeRobot datasets
DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")

# Root directory of the source LBM data (where tasks/ exists)
LBM_ROOT = Path("/nfs_gaoyang/LBM_dataset")

# Set HF_LEROBOT_HOME to ensure split datasets are saved in the same root
os.environ["HF_LEROBOT_HOME"] = str(DATASET_ROOT)

# ==============================================================================
# Worker Function
# ==============================================================================

def process_skill(skill_name: str):
    """
    Process a single skill: build mapping, read splits, and call edit script.
    """
    try:
        logging.info(f"--- Processing skill: {skill_name} ---")
        
        # 1. Establish mapping from source path to episode index
        # This must exactly match the logic used during initial conversion.
        # Based on lbm_dataset.py, episodes are sorted lexicographically by their full path.
        skill_source_root = LBM_ROOT / "tasks" / skill_name
        if not skill_source_root.exists():
            logging.error(f"Source folder not found for skill {skill_name}: {skill_source_root}")
            return

        # Pattern matches the one in lbm_dataset.py:_discover_episode_dirs
        pattern = f"tasks/{skill_name}/**/diffusion_spartan/episode_*/processed"
        all_episodes = sorted(LBM_ROOT.glob(pattern))
        
        if not all_episodes:
            logging.warning(f"No source episodes discovered for skill {skill_name} using pattern: {pattern}")
            return
            
        # Split files contain paths relative to tasks/{skill_name}/
        # e.g. "riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_1/processed"
        path_to_idx = {str(p.relative_to(skill_source_root)): i for i, p in enumerate(all_episodes)}
        
        # 2. Identify the target LeRobot dataset
        # Default convention: lbm-eval-skills-<skill_name_lowercase>
        repo_id = f"lbm-eval-skills-{skill_name.lower()}"
        dataset_path = DATASET_ROOT / repo_id
        
        if not dataset_path.exists():
            logging.warning(f"Target dataset directory not found: {dataset_path}. Skipping skill {skill_name}.")
            return
            
        # Check for meta/info.json to ensure it's a valid LeRobot dataset and avoid hub errors
        if not (dataset_path / "meta" / "info.json").exists():
            logging.warning(f"Dataset {repo_id} exists but is missing meta/info.json. Skipping.")
            return

        # 3. Read split files and collect indices
        split_dir = SPLIT_FILE_ROOT / skill_name
        splits = {}
        split_files = list(split_dir.glob("*_episode_dirs.txt"))
        
        if not split_files:
            logging.warning(f"No split files (*_episode_dirs.txt) found in {split_dir}")
            return

        for split_file in split_files:
            # e.g., train_episode_dirs.txt -> split name 'train'
            split_name = split_file.name.replace("_episode_dirs.txt", "")
            
            with open(split_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            indices = []
            for line in lines:
                if line in path_to_idx:
                    indices.append(path_to_idx[line])
                else:
                    # Robust check for leading/trailing slashes
                    line_clean = line.strip("/")
                    if line_clean in path_to_idx:
                        indices.append(path_to_idx[line_clean])
                    else:
                        logging.warning(f"[{skill_name}] Path not found in discovered episodes: {line}")
            
            if indices:
                # Store sorted indices for the split
                splits[split_name] = sorted(indices)

        if not splits:
            logging.warning(f"No valid indices found for any split in {skill_name}. Skipping.")
            return

        # 4. Invoke the edit script to perform the actual split
        # We use 'python -m' to run the script in the current environment.
        # --root is provided so it finds the existing dataset in DATASET_ROOT.
        
        # Ensure PYTHONPATH includes the 'src' directory so 'lerobot' can be found
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        src_path = project_root / "src"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(src_path)

        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset",
            "--repo_id", repo_id,
            "--root", str(dataset_path),
            "--operation.type", "split",
            "--operation.splits", json.dumps(splits)
        ]
        
        logging.info(f"[{skill_name}] Running command: {' '.join(cmd)}")
        
        # Run subprocess and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logging.error(f"[{skill_name}] Command failed with return code {result.returncode}")
            logging.error(f"[{skill_name}] STDERR:\n{result.stderr}")
            # Log first few lines of stdout if it's long
            stdout_lines = result.stdout.splitlines()
            if stdout_lines:
                logging.error(f"[{skill_name}] STDOUT (truncated):\n" + "\n".join(stdout_lines[:20]))
            # record the failed repo_id 
            with open("failed_split_lbm_eval_skills_repo_ids.txt", "a") as f:
                f.write(f"{repo_id}\n")
        else:
            logging.info(f"[{skill_name}] Successfully created splits: {list(splits.keys())}")
            
    except Exception as e:
        logging.exception(f"Unexpected error processing skill '{skill_name}': {e}")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Setup basic logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if not SPLIT_FILE_ROOT.exists():
        logging.error(f"Split file root directory does not exist: {SPLIT_FILE_ROOT}")
        sys.exit(1)

    # Discover all skill directories
    skill_dirs = [d for d in SPLIT_FILE_ROOT.iterdir() if d.is_dir()]
    skills = sorted([d.name for d in skill_dirs])
    
    if not skills:
        logging.info("No skill directories found in splits path. Nothing to do.")
        return

    logging.info(f"Found {len(skills)} skills to process in {SPLIT_FILE_ROOT}")
    logging.info(f"Datasets will be loaded from and saved to: {DATASET_ROOT}")
    
    # Process skills using a process pool for efficiency
    # The number of workers is limited by the number of skills and available CPU cores.
    # num_workers = min(len(skills), os.cpu_count() or 4)
    num_workers = 12
    logging.info(f"Initializing process pool with {num_workers} workers.")
    
    with Pool(processes=num_workers) as pool:
        pool.map(process_skill, skills)

    logging.info("Partitioning process completed.")

if __name__ == "__main__":
    main()

