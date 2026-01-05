#!/usr/bin/env python3

import os
import json
import logging
import shutil
from pathlib import Path
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

# Failed datasets list
FAILED_DATASETS_FILE = Path("failed_split_lbm_eval_skills_repo_ids_old.txt")

# Common split suffixes (based on split file naming convention)
SPLIT_SUFFIXES = ["_train", "_iid", "_ood_manipuland", "_ood_operator", "_ood_skill"]

# Set HF_LEROBOT_HOME
os.environ["HF_LEROBOT_HOME"] = str(DATASET_ROOT)

# skill name list
SKILL_NAME_LIST = [
    "BimanualPlaceAppleFromBowlIntoBin",
    "BimanualPlacePearFromBowlIntoBin",
    "BimanualPlaceFruitFromBowlIntoBin",
    "BimanualPlaceAvocadoFromBowlIntoBin",
    "BimanualPlaceFruitFromBowlOnCuttingBoard",
    "BimanualPlaceAvocadoFromBowlOnCuttingBoard",
    "BimanualPlacePearFromBowlOnCuttingBoard",
    "BimanualPlaceAppleFromBowlOnCuttingBoard",
    "BimanualHangMugsOnMugHolderFromDryingRack",
    "BimanualHangMugsOnMugHolderFromTable",
    "BimanualPutSpatulaOnPlateFromDryingRack",
    "BimanualPutSpatulaOnPlateFromTable",
    "BimanualPutSpatulaOnPlateFromUtensilCrock",
    "BimanualPutSpatulaOnTableFromDryingRack",
    "BimanualPutSpatulaOnTableFromUtensilCrock",
    "BimanualStackPlatesOnTableFromDryingRack",
    "BimanualStackPlatesOnTableFromTable",
    "BimanualStoreCerealBoxUnderShelf",
    "DumpVegetablesFromSmallToLargeContainer",
    "PickAndPlaceBox",
    "PlaceCupByCoaster",
    "PlaceCupOnCoaster",
    "PushCoasterToCenterOfTable",
    "PushCoasterToMug",
    "PutBananaInCenterOfTable",
    "PutBananaOnSaucer",
    "PutContainersOnPlate",
    "PutCupInCenterOfTable",
    "PutCupOnSaucer",
    "PutFruitInLargeContainerAndCoverWithPlate",
    "PutGreenAppleInCenterOfTable",
    "PutGreenAppleOnSaucer",
    "PutKiwiInCenterOfTable",
    "PutKiwiOnSaucer",
    "PutMugInCenterOfTable",
    "PutMugOnSaucer",
    "PutOrangeInCenterOfTable",
    "PutOrangeOnSaucer",
    "PutSpatulaInUtensilCrock",
    "PutSpatulaInUtensilCrockFromDryingRack",
    "SeparateFruitsVegetablesIntoContainers",
]

# ==============================================================================
# Cleanup and Retry Logic
# ==============================================================================

def cleanup_failed_splits(repo_id: str):
    """
    Clean up all split datasets for a given repo_id.
    """
    cleaned = []
    for suffix in SPLIT_SUFFIXES:
        split_repo_id = f"{repo_id}{suffix}"
        split_path = DATASET_ROOT / split_repo_id
        if split_path.exists():
            logging.info(f"  Removing {split_repo_id}...")
            shutil.rmtree(split_path)
            cleaned.append(split_repo_id)
    return cleaned

def retry_split(repo_id: str):
    """
    Retry splitting a single dataset.
    """
    try:
        # Extract skill name from repo_id (e.g., lbm-eval-skills-xxx -> xxx)
        skill_name = repo_id.replace("lbm-eval-skills-", "")
        for ref_skill_name in SKILL_NAME_LIST:
            if ref_skill_name.lower() == skill_name.lower():
                skill_name = ref_skill_name
                break
        else:
            logging.error(f"Skill name not found: {skill_name}")
            return False
        
        logging.info(f"--- Retrying split for: {repo_id} (skill: {skill_name}) ---")
        
        # 1. Establish mapping from source path to episode index
        skill_source_root = LBM_ROOT / "tasks" / skill_name
        if not skill_source_root.exists():
            logging.error(f"Source folder not found: {skill_source_root}")
            return False

        pattern = f"tasks/{skill_name}/**/diffusion_spartan/episode_*/processed"
        all_episodes = sorted(LBM_ROOT.glob(pattern))
        
        if not all_episodes:
            logging.warning(f"No source episodes found for {skill_name}")
            return False
            
        path_to_idx = {str(p.relative_to(skill_source_root)): i for i, p in enumerate(all_episodes)}
        
        # 2. Check dataset exists
        dataset_path = DATASET_ROOT / repo_id
        if not dataset_path.exists() or not (dataset_path / "meta" / "info.json").exists():
            logging.error(f"Dataset not found: {dataset_path}")
            return False

        # 3. Read split files
        split_dir = SPLIT_FILE_ROOT / skill_name
        splits = {}
        split_files = list(split_dir.glob("*_episode_dirs.txt"))
        
        if not split_files:
            logging.warning(f"No split files found in {split_dir}")
            return False

        for split_file in split_files:
            split_name = split_file.name.replace("_episode_dirs.txt", "")
            
            with open(split_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            indices = []
            for line in lines:
                if line in path_to_idx:
                    indices.append(path_to_idx[line])
                else:
                    line_clean = line.strip("/")
                    if line_clean in path_to_idx:
                        indices.append(path_to_idx[line_clean])
                    else:
                        logging.warning(f"Path not found: {line}")
            
            if indices:
                splits[split_name] = sorted(indices)

        if not splits:
            logging.warning(f"No valid splits found for {skill_name}")
            return False

        # 4. Invoke split command
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
        
        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logging.error(f"Split failed for {repo_id}")
            logging.error(f"STDERR:\n{result.stderr}")
            return False
        else:
            logging.info(f"✓ Successfully split {repo_id} into: {list(splits.keys())}")
            return True
            
    except Exception as e:
        logging.exception(f"Error retrying split for {repo_id}: {e}")
        return False

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if not FAILED_DATASETS_FILE.exists():
        logging.error(f"Failed datasets file not found: {FAILED_DATASETS_FILE}")
        sys.exit(1)
    
    # Read failed dataset IDs
    with open(FAILED_DATASETS_FILE, "r") as f:
        failed_repo_ids = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Found {len(failed_repo_ids)} failed datasets to retry")
    logging.info("=" * 80)
    
    # Step 1: Clean up existing split datasets
    logging.info("Step 1: Cleaning up existing split datasets...")
    all_cleaned = []
    for repo_id in failed_repo_ids:
        logging.info(f"Cleaning splits for {repo_id}:")
        cleaned = cleanup_failed_splits(repo_id)
        if cleaned:
            all_cleaned.extend(cleaned)
        else:
            logging.info(f"  No existing splits found for {repo_id}")
    
    logging.info(f"Cleaned up {len(all_cleaned)} split datasets: {all_cleaned}")
    logging.info("=" * 80)
    
    # Step 2: Retry splitting
    logging.info("Step 2: Retrying split operations...")
    success_count = 0
    failed_retries = []
    
    for repo_id in failed_repo_ids:
        if retry_split(repo_id):
            success_count += 1
        else:
            failed_retries.append(repo_id)
        logging.info("-" * 80)
    
    # Summary
    logging.info("=" * 80)
    logging.info("Retry Summary:")
    logging.info(f"  Total datasets: {len(failed_repo_ids)}")
    logging.info(f"  Successfully split: {success_count}")
    logging.info(f"  Still failed: {len(failed_retries)}")
    
    if failed_retries:
        logging.info("\nStill failed datasets:")
        for repo_id in failed_retries:
            logging.info(f"  - {repo_id}")
        # Write new failed list
        with open("failed_split_lbm_eval_skills_repo_ids_new.txt", "w") as f:
            for repo_id in failed_retries:
                f.write(f"{repo_id}\n")
        logging.info(f"\nNew failed list written to: failed_split_lbm_eval_skills_repo_ids_new.txt")

if __name__ == "__main__":
    main()

