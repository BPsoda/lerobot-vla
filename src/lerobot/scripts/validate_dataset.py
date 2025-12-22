#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")
DEFAULT_MERGE_PLAN_FILE = Path("/nfs_gaoyang/LBM_lerobot_dataset/dataset_lists/merge_plan_iid.txt")

# ==============================================================================
# Validation Logic
# ==============================================================================

def validate_dataset(repo_id: str, dataset_root: Path, tolerance_s: float = 0.2) -> tuple[bool, str]:
    """
    Validate a single dataset by attempting to load and iterate through all frames.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        dataset_path = dataset_root / repo_id
        if not dataset_path.exists():
            return False, f"Dataset directory does not exist: {dataset_path}"
        
        if not (dataset_path / "meta" / "info.json").exists():
            return False, f"Missing meta/info.json"
        
        # Try to load the dataset
        dataset = LeRobotDataset(repo_id, root=dataset_root, tolerance_s=tolerance_s)
        
        if dataset.num_frames == 0:
            return False, "Dataset has 0 frames"
        
        # Try to access first frame to check basic structure
        try:
            sample = dataset[0]
            if not isinstance(sample, dict):
                return False, f"Sample is not a dict, got {type(sample)}"
        except Exception as e:
            return False, f"Failed to load first frame: {e}"
        
        # Try to access last frame
        try:
            last_frame = dataset[dataset.num_frames - 1]
        except Exception as e:
            return False, f"Failed to load last frame (idx {dataset.num_frames - 1}): {e}"
        
        # Try to iterate through all frames (with progress bar)
        failed_indices = []
        for idx in tqdm(range(dataset.num_frames), desc=f"Validating {repo_id}", leave=False):
            try:
                frame = dataset[idx]
                # Basic sanity check: frame should be a dict
                if not isinstance(frame, dict):
                    failed_indices.append((idx, f"Frame is not a dict"))
            except Exception as e:
                failed_indices.append((idx, str(e)))
                # Stop after first few errors to avoid flooding
                if len(failed_indices) >= 10:
                    break
        
        if failed_indices:
            error_msg = f"Found {len(failed_indices)} invalid frames. First few: {failed_indices[:5]}"
            return False, error_msg
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Exception during validation: {e}"

def main():
    parser = argparse.ArgumentParser(description="Validate datasets from a merge plan file")
    parser.add_argument("--merge-plan-file", type=str, default=str(DEFAULT_MERGE_PLAN_FILE),
                        help="Path to merge plan txt file (comma-separated dataset IDs)")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT),
                        help="Root directory containing datasets")
    parser.add_argument("--tolerance-s", type=float, default=0.2,
                        help="Tolerance for timestamp matching")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    merge_plan_file = Path(args.merge_plan_file)
    dataset_root = Path(args.dataset_root)
    
    if not merge_plan_file.exists():
        logging.error(f"Merge plan file not found: {merge_plan_file}")
        return
    
    # Read dataset IDs from file
    with open(merge_plan_file, "r") as f:
        content = f.read().strip()
        if not content:
            logging.error("Merge plan file is empty")
            return
        
        # Split by comma and clean up whitespace
        repo_ids = [repo_id.strip() for repo_id in content.split(",") if repo_id.strip()]
    
    logging.info(f"Found {len(repo_ids)} datasets to validate")
    logging.info(f"Dataset root: {dataset_root}")
    
    # Validate each dataset
    results = []
    for repo_id in tqdm(repo_ids, desc="Validating datasets"):
        is_valid, message = validate_dataset(repo_id, dataset_root, tolerance_s=args.tolerance_s)
        results.append((repo_id, is_valid, message))
        
        if is_valid:
            logging.info(f"✓ {repo_id}: {message}")
        else:
            logging.error(f"✗ {repo_id}: {message}")
    
    # Summary
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    invalid_count = len(results) - valid_count
    
    logging.info("=" * 80)
    logging.info("Validation Summary:")
    logging.info(f"  Total datasets: {len(results)}")
    logging.info(f"  Valid: {valid_count}")
    logging.info(f"  Invalid: {invalid_count}")
    
    if invalid_count > 0:
        logging.info("\nInvalid datasets:")
        for repo_id, is_valid, message in results:
            if not is_valid:
                logging.info(f"  - {repo_id}: {message}")
        return 1
    else:
        logging.info("All datasets are valid!")
        return 0

if __name__ == "__main__":
    exit(main())
