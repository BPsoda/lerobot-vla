#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")
DEFAULT_MERGE_PLAN_FILE = Path("/nfs_gaoyang/LBM_lerobot_dataset/dataset_lists/merge_plan_iid.txt")
DEFAULT_LOG_FILE = "validate.out"

# ==============================================================================
# Validation Logic
# ==============================================================================

def validate_dataset(repo_id: str, dataset_root: str, tolerance_s: float = 0.2) -> tuple[bool, str]:
    """
    Validate a single dataset by attempting to load and iterate through all frames.
    This function is designed to be called in parallel.
    
    Args:
        repo_id: Dataset repository ID
        dataset_root: Path to dataset root directory (as string for pickling)
        tolerance_s: Tolerance for timestamp matching
    
    Returns:
        (repo_id, is_valid, error_message)
    """
    try:
        dataset_root_path = Path(dataset_root)
        dataset_path = dataset_root_path / repo_id
        if not dataset_path.exists():
            return (repo_id, False, f"Dataset directory does not exist: {dataset_path}")
        
        if not (dataset_path / "meta" / "info.json").exists():
            return (repo_id, False, f"Missing meta/info.json")
        
        # Try to load the dataset
        dataset = LeRobotDataset(repo_id, root=dataset_root_path, tolerance_s=tolerance_s)
        
        if dataset.num_frames == 0:
            return (repo_id, False, "Dataset has 0 frames")
        
        # Try to access first frame to check basic structure
        try:
            sample = dataset[0]
            if not isinstance(sample, dict):
                return (repo_id, False, f"Sample is not a dict, got {type(sample)}")
        except Exception as e:
            return (repo_id, False, f"Failed to load first frame: {e}")
        
        # Try to access last frame
        try:
            last_frame = dataset[dataset.num_frames - 1]
        except Exception as e:
            return (repo_id, False, f"Failed to load last frame (idx {dataset.num_frames - 1}): {e}")
        
        # Try to iterate through all frames (without tqdm for parallel execution)
        failed_indices = []
        for idx in range(dataset.num_frames):
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
            return (repo_id, False, error_msg)
        
        return (repo_id, True, "OK")
        
    except Exception as e:
        return (repo_id, False, f"Exception during validation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate datasets from a merge plan file")
    parser.add_argument("--merge-plan-file", type=str, default=str(DEFAULT_MERGE_PLAN_FILE),
                        help="Path to merge plan txt file (comma-separated dataset IDs)")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT),
                        help="Root directory containing datasets")
    parser.add_argument("--tolerance-s", type=float, default=0.2,
                        help="Tolerance for timestamp matching")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE,
                        help="Path to log file (default: validate.out)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of parallel workers (default: min(4, num_datasets))")
    args = parser.parse_args()
    
    # Setup logging to both console and file
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler
    log_file_path = Path(args.log_file)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    logging.info(f"Logging to file: {log_file_path}")
    
    merge_plan_file = Path(args.merge_plan_file)
    dataset_root = Path(args.dataset_root)
    
    if not merge_plan_file.exists():
        logging.error(f"Merge plan file not found: {merge_plan_file}")
        return 1
    
    # Read dataset IDs from file
    with open(merge_plan_file, "r") as f:
        content = f.read().strip()
        if not content:
            logging.error("Merge plan file is empty")
            return 1
        
        # Split by comma and clean up whitespace
        repo_ids = [repo_id.strip() for repo_id in content.split(",") if repo_id.strip()]
    
    logging.info(f"Found {len(repo_ids)} datasets to validate")
    logging.info(f"Dataset root: {dataset_root}")
    
    # Determine number of workers (limit to prevent OOM)
    if args.max_workers is None:
        max_workers = min(4, len(repo_ids), (os.cpu_count() or 4) // 2)
    else:
        max_workers = args.max_workers
    
    logging.info(f"Using {max_workers} parallel workers for validation")
    
    # Validate each dataset in parallel
    results = []
    validate_func = partial(validate_dataset, dataset_root=str(dataset_root), tolerance_s=args.tolerance_s)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_repo = {executor.submit(validate_func, repo_id): repo_id for repo_id in repo_ids}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_repo), total=len(repo_ids), desc="Validating datasets"):
            repo_id, is_valid, message = future.result()
            results.append((repo_id, is_valid, message))
            
            if is_valid:
                logging.info(f"✓ {repo_id}: {message}")
            else:
                logging.error(f"✗ {repo_id}: {message}")
    
    # Sort results by original order for consistent output
    results_dict = {repo_id: (is_valid, message) for repo_id, is_valid, message in results}
    results = [(repo_id, results_dict[repo_id][0], results_dict[repo_id][1]) for repo_id in repo_ids]
    
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
        logging.info(f"\nFull log saved to: {log_file_path}")
        return 1
    else:
        logging.info("All datasets are valid!")
        logging.info(f"Full log saved to: {log_file_path}")
        return 0

if __name__ == "__main__":
    exit(main())
