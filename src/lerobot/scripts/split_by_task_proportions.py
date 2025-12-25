#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import shutil
from pathlib import Path
import subprocess
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")

# ==============================================================================
# Split by Task Proportions
# ==============================================================================

def get_episodes_by_task(dataset: LeRobotDataset) -> dict[str, list[int]]:
    """
    Group episode indices by task.
    Episodes are kept in their original order (by episode_index).
    
    Returns:
        dict mapping task name to list of episode indices (sorted)
    """
    episodes_by_task = {}
    
    # Get all episodes and their tasks
    for ep_idx in range(dataset.num_episodes):
        # Get task for this episode by checking the first frame
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        first_frame = dataset.hf_dataset[from_idx]
        task_idx = first_frame["task_index"].item() if hasattr(first_frame["task_index"], "item") else first_frame["task_index"]
        task_name = dataset.meta.tasks.iloc[task_idx].name
        
        if task_name not in episodes_by_task:
            episodes_by_task[task_name] = []
        episodes_by_task[task_name].append(ep_idx)
    
    # Ensure episodes are sorted by index for each task
    for task_name in episodes_by_task:
        episodes_by_task[task_name].sort()
    
    return episodes_by_task

def split_by_task_proportions(
    repo_id: str,
    dataset_root: Path,
    proportions: list[float],
    output_suffixes: list[str] | None = None
) -> None:
    """
    Split a dataset by task proportions.
    
    For each task, take the first a1, a2, a3... proportions of episodes,
    and create new datasets for each proportion.
    
    Args:
        repo_id: Source dataset repository ID
        dataset_root: Root directory containing datasets
        proportions: List of proportions (e.g., [0.5, 0.3, 0.2])
        output_suffixes: Optional list of suffixes for output datasets (default: ["_p1", "_p2", ...])
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    
    # Group episodes by task
    logging.info("Grouping episodes by task...")
    episodes_by_task = get_episodes_by_task(dataset)
    
    logging.info(f"Found {len(episodes_by_task)} tasks:")
    for task_name, ep_indices in episodes_by_task.items():
        logging.info(f"  {task_name}: {len(ep_indices)} episodes")
    
    # Generate output suffixes if not provided
    if output_suffixes is None:
        output_suffixes = [f"_p{i+1}" for i in range(len(proportions))]
    else:
        # Ensure suffixes start with underscore if not already
        output_suffixes = [f"_{s}" if not s.startswith("_") else s for s in output_suffixes]
    
    if len(proportions) != len(output_suffixes):
        raise ValueError(f"Number of proportions ({len(proportions)}) must match number of suffixes ({len(output_suffixes)})")
    
    # For each proportion, collect episodes from each task
    for prop_idx, (proportion, suffix) in enumerate(zip(proportions, output_suffixes)):
        logging.info(f"\n--- Creating split {suffix} (proportion: {proportion}) ---")
        
        selected_episodes = []
        for task_name, ep_indices in episodes_by_task.items():
            # Calculate number of episodes to take for this task
            num_episodes = len(ep_indices)
            num_to_take = max(1, int(num_episodes * proportion))
            
            # Take first num_to_take episodes
            selected = ep_indices[:num_to_take]
            selected_episodes.extend(selected)
            
            logging.info(f"  {task_name}: {num_to_take}/{num_episodes} episodes")
        
        selected_episodes = sorted(selected_episodes)
        logging.info(f"Total selected episodes: {len(selected_episodes)}")
        
        # Create split using lerobot_edit_dataset
        output_repo_id = f"{repo_id}{suffix}"
        
        # Prepare splits dict (only one split in this case)
        splits = {"main": selected_episodes}
        
        # Setup environment
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
            "--root", str(dataset_root),
            "--operation.type", "split",
            "--operation.splits", json.dumps(splits)
        ]
        
        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logging.error(f"Failed to create split {suffix}")
            logging.error(f"STDERR:\n{result.stderr}")
            if result.stdout:
                logging.error(f"STDOUT:\n{result.stdout[:500]}")
        else:
            # Rename the output dataset
            split_output_dir = dataset_root / f"{repo_id}_main"
            target_output_dir = dataset_root / output_repo_id
            if split_output_dir.exists():
                if target_output_dir.exists():
                    shutil.rmtree(target_output_dir)
                shutil.move(str(split_output_dir), str(target_output_dir))
                logging.info(f"✓ Created dataset: {output_repo_id}")
            else:
                logging.warning(f"Split output directory not found: {split_output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Split dataset by task proportions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into 3 datasets with proportions 0.5, 0.3, 0.2
  python split_by_task_proportions.py \\
      --repo-id lbm-eval-train \\
      --proportions 0.5 0.3 0.2

  # Custom output suffixes
  python split_by_task_proportions.py \\
      --repo-id lbm-eval-train \\
      --proportions 0.5 0.3 0.2 \\
      --suffixes small medium large
        """
    )
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Source dataset repository ID")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT),
                        help="Root directory containing datasets")
    parser.add_argument("--proportions", type=float, nargs="+", required=True,
                        help="List of proportions (e.g., 0.5 0.3 0.2)")
    parser.add_argument("--suffixes", type=str, nargs="+", default=None,
                        help="Optional list of suffixes for output datasets (default: _p1, _p2, ...)")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Validate proportions
    for prop in args.proportions:
        if prop <= 0 or prop > 1:
            logging.error(f"Proportions must be between 0 and 1, got: {prop}")
            return 1
    
    dataset_root = Path(args.dataset_root)
    
    try:
        split_by_task_proportions(
            repo_id=args.repo_id,
            dataset_root=dataset_root,
            proportions=args.proportions,
            output_suffixes=args.suffixes
        )
        logging.info("\nAll splits created successfully!")
        return 0
    except Exception as e:
        logging.exception(f"Error during split: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

