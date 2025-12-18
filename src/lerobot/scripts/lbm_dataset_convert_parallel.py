"""
Convert LBM diffusion-spartan dataset to LeRobotDataset format with parallel processing.

This script parallelizes the conversion by processing each skill_type in a separate process,
creating independent datasets for each skill_type. This significantly speeds up conversion
when multiple skill_types are specified.

Usage:
    python lbm_dataset_convert_parallel.py \
        --lbm_root /path/to/lbm_root \
        --output_root /path/to/output_root \
        --repo_id lbm-eval \
        --fps 10 \
        --use_videos True \
        --keep_cameras None \
        --state_keys "robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__grippers__left::panda_hand,robot__actual__grippers__right::panda_hand" \
        --skill_types "skill1,skill2,skill3" \
        --num_workers 4 \
        --merge_datasets False
"""

from lerobot.datasets.lbm_dataset import LBMDatasetConverter
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from argparse import ArgumentParser
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import cProfile
import pstats
from contextlib import contextmanager
import re
import shutil

logging.basicConfig(level=logging.INFO)


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for argparse.
    
    Args:
        value: String value to convert
        
    Returns:
        Boolean value
        
    Raises:
        argparse.ArgumentTypeError: If value is not a valid boolean string
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {value}')


def sanitize_skill_type(skill_type: str) -> str:
    """Convert skill_type name into a safe identifier for repo_id.
    
    Removes or replaces special characters that might cause issues in file paths
    or repository names.
    
    Args:
        skill_type: Raw skill type name
        
    Returns:
        Sanitized skill type name safe for use in repo_id
    """
    # Replace spaces and hyphens with underscores
    safe = skill_type.replace(" ", "_").replace("-", "_")
    # Remove any remaining special characters except underscores
    safe = re.sub(r'[^a-zA-Z0-9_]', '', safe)
    # Convert to lowercase
    safe = safe.lower()
    # Remove consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    return safe if safe else "unknown"


@contextmanager
def profile_context(enable_profile, output_file=None):
    """Context manager for performance profiling."""
    if enable_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        yield profiler
        profiler.disable()
        
        # Create stats object
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        if output_file:
            # Save to file
            stats.dump_stats(output_file)
            logging.info(f"Profile results saved to {output_file}")
        else:
            # Print to stdout
            stats.print_stats(50)  # Print top 50 functions
    else:
        yield None


def convert_single_skill_type(
    lbm_root: str,
    output_root: str,
    base_repo_id: str,
    skill_type: str,
    fps: int,
    use_videos: bool,
    keep_cameras: Optional[list[str]],
    state_keys: Optional[list[str]],
    overwrite: bool,
) -> tuple[str, Path]:
    """Convert a single skill_type to a LeRobotDataset.
    
    Args:
        lbm_root: Root directory of LBM dataset
        output_root: Output root directory
        base_repo_id: Base repository ID (will append skill_type)
        skill_type: Skill type to process
        fps: Frames per second
        use_videos: Whether to use videos
        keep_cameras: List of cameras to keep
        state_keys: List of state keys to include
        overwrite: Whether to overwrite existing dataset
        
    Returns:
        Tuple of (repo_id, dataset_path) for the converted dataset
    """
    # Create repo_id with sanitized skill_type suffix
    sanitized_skill = sanitize_skill_type(skill_type)
    repo_id = f"{base_repo_id}-{sanitized_skill}"
    
    logging.info(f"[{skill_type}] Starting conversion to {repo_id}")
    
    # Check if dataset exists and handle overwrite
    ds_root = Path(output_root) / repo_id
    if overwrite and ds_root.exists():
        logging.warning(f"[{skill_type}] Overwriting existing dataset at {ds_root}")
        shutil.rmtree(ds_root)
    
    try:
        converter = LBMDatasetConverter(
            lbm_root=lbm_root,
            output_root=output_root,
            repo_id=repo_id,
            fps=fps,
            use_videos=use_videos,
            keep_cameras=keep_cameras,
            state_keys=state_keys,
            skill_types=[skill_type],  # Only process this skill_type
        )
        
        # Convert the dataset
        dataset = converter.convert()
        
        logging.info(
            f"[{skill_type}] Conversion complete: {dataset.meta.total_episodes} episodes, "
            f"{dataset.meta.total_frames} frames"
        )
        
        return repo_id, dataset.root
        
    except Exception as e:
        logging.error(f"[{skill_type}] Conversion failed: {e}")
        raise


def main():
    parser = ArgumentParser()
    parser.add_argument("--lbm_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--use_videos", type=str_to_bool, default=True,
                       help="Whether to use videos (True/False)")
    parser.add_argument("--keep_cameras", type=str, default=None)
    parser.add_argument("--state_keys", type=str, default=None)
    parser.add_argument("--skill_types", type=str, required=True, 
                       help="Comma-separated list of skill types to process")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: number of skill_types)")
    parser.add_argument("--merge_datasets", action="store_true",
                       help="Merge all skill_type datasets into a single dataset")
    parser.add_argument("--merged_repo_id", type=str, default=None,
                       help="Repository ID for merged dataset (default: {repo_id}-merged)")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--profile_output", type=str, default=None, 
                       help="Output file for profile results (default: print to stdout)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing datasets instead of resuming")

    args = parser.parse_args()
    
    # Report config
    logging.info(f"LBM root: {args.lbm_root}")
    logging.info(f"Output root: {args.output_root}")
    logging.info(f"Base repo ID: {args.repo_id}")
    logging.info(f"FPS: {args.fps}")
    logging.info(f"Use videos: {args.use_videos}")
    logging.info(f"Skill types: {args.skill_types}")
    if args.keep_cameras is not None:
        logging.info(f"Keep cameras: {args.keep_cameras.split(',')}")
    if args.state_keys is not None:
        logging.info(f"State keys: {args.state_keys.split(',')}")
    logging.info(f"Number of workers: {args.num_workers}")
    logging.info(f"Merge datasets: {args.merge_datasets}")
    print("--------------------------------")

    # Parse skill_types
    skill_types = [s.strip() for s in args.skill_types.split(',') if s.strip()]
    if not skill_types:
        raise ValueError("At least one skill_type must be specified")
    
    # Check for duplicates after sanitization
    sanitized_skills = [sanitize_skill_type(st) for st in skill_types]
    if len(sanitized_skills) != len(set(sanitized_skills)):
        duplicates = {}
        for orig, sanitized in zip(skill_types, sanitized_skills):
            if sanitized in duplicates:
                duplicates[sanitized].append(orig)
            else:
                duplicates[sanitized] = [orig]
        duplicate_pairs = {k: v for k, v in duplicates.items() if len(v) > 1}
        raise ValueError(
            f"Duplicate repo_ids will be generated after sanitization:\n"
            f"{duplicate_pairs}\n"
            f"Please ensure skill_types have unique names after sanitization."
        )
    
    # Report sanitized names
    logging.info("Skill types (original -> sanitized):")
    for orig, sanitized in zip(skill_types, sanitized_skills):
        if orig != sanitized:
            logging.info(f"  '{orig}' -> '{sanitized}'")
        else:
            logging.info(f"  '{orig}'")
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else len(skill_types)
    num_workers = min(num_workers, len(skill_types))  # Don't use more workers than skill_types
    
    # Parse optional arguments
    keep_cameras = args.keep_cameras.split(',') if args.keep_cameras is not None else None
    state_keys = args.state_keys.split(',') if args.state_keys is not None else None
    
    logging.info(f"Processing {len(skill_types)} skill types with {num_workers} parallel workers")

    # Prepare arguments for each worker
    convert_args = [
        (
            args.lbm_root,
            args.output_root,
            args.repo_id,
            skill_type,
            args.fps,
            args.use_videos,
            keep_cameras,
            state_keys,
            args.overwrite,
        )
        for skill_type in skill_types
    ]

    # Process skill_types in parallel
    with profile_context(args.profile, args.profile_output):
        if num_workers == 1:
            # Sequential processing (useful for debugging)
            results = []
            for skill_type, convert_arg in zip(skill_types, convert_args):
                result = convert_single_skill_type(*convert_arg)
                results.append((skill_type, result))
        else:
            # Parallel processing
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_skill = {
                    executor.submit(convert_single_skill_type, *convert_arg): skill_type
                    for skill_type, convert_arg in zip(skill_types, convert_args)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_skill):
                    skill_type = future_to_skill[future]
                    try:
                        result = future.result()
                        results.append((skill_type, result))
                        logging.info(f"[{skill_type}] ✓ Completed successfully")
                    except Exception as e:
                        logging.error(f"[{skill_type}] ✗ Failed: {e}")
                        raise

    # Report results
    logging.info("\n" + "="*60)
    logging.info("Conversion Summary:")
    logging.info("="*60)
    for skill_type, (repo_id, dataset_path) in results:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        logging.info(
            f"  {skill_type}: {repo_id} - "
            f"{dataset.meta.total_episodes} episodes, "
            f"{dataset.meta.total_frames} frames"
        )
    logging.info("="*60)

    # Optionally merge datasets
    if args.merge_datasets:
        merged_repo_id = args.merged_repo_id or f"{args.repo_id}-merged"
        logging.info(f"\nMerging {len(results)} datasets into {merged_repo_id}")
        
        repo_ids = [repo_id for _, (repo_id, _) in results]
        # roots should be the output_root directory (parent of each repo_id directory)
        # All datasets are under the same output_root
        roots = [Path(args.output_root)] * len(repo_ids)
        merged_output_root = Path(args.output_root) / merged_repo_id
        
        try:
            aggregate_datasets(
                repo_ids=repo_ids,
                aggr_repo_id=merged_repo_id,
                roots=roots,
                aggr_root=merged_output_root,
            )
            
            merged_dataset = LeRobotDataset(repo_id=merged_repo_id, root=merged_output_root)
            logging.info(
                f"\n✓ Merged dataset created: {merged_repo_id}\n"
                f"  Total episodes: {merged_dataset.meta.total_episodes}\n"
                f"  Total frames: {merged_dataset.meta.total_frames}\n"
                f"  Location: {merged_output_root}"
            )
        except Exception as e:
            logging.error(f"Failed to merge datasets: {e}")
            raise
    else:
        logging.info(
            f"\nTo merge all datasets, run with --merge_datasets flag:\n"
            f"  python -m lerobot.scripts.lbm_dataset_convert_parallel \\\n"
            f"    --merge_datasets \\\n"
            f"    --merged_repo_id {args.repo_id}-merged \\\n"
            f"    ... (other arguments)"
        )


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Linux systems
    multiprocessing.set_start_method('spawn', force=True)
    main()

