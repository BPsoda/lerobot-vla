"""
Convert LBM diffusion-spartan dataset to LeRobotDataset format.

Usage:
    python lbm_dataset_convert.py \
        --lbm_root /path/to/lbm_root \
        --output_root /path/to/output_root \
        --repo_id lbm-eval \
        --fps 10 \
        --use_videos True \
        --keep_cameras scene_right_0,scene_left_0,wrist_right_minus,wrist_left_plus \
        --state_keys "robot__actual__poses__right::panda__xyz,robot__actual__poses__right::panda__rot_6d,robot__actual__poses__left::panda__xyz,robot__actual__poses__left::panda__rot_6d,robot__actual__grippers__right::panda_hand,robot__actual__grippers__left::panda_hand"
"""

from lerobot.common.datasets.lbm_dataset import LBMDatasetConverter
from argparse import ArgumentParser
import logging
import cProfile
import pstats
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)


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


parser = ArgumentParser()
parser.add_argument("--lbm_root", type=str, required=True)
parser.add_argument("--output_root", type=str, required=True)
parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--use_videos", type=bool, default=True)
parser.add_argument("--keep_cameras", type=str, default=None)
parser.add_argument("--state_keys", type=str, default=None)
parser.add_argument("--skill_types", type=str, default=None)
parser.add_argument("--episode_dir_list", type=str, default=None, help="Path to a file containing episode directories (one per line). If not specified, episodes will be discovered automatically.")
parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
parser.add_argument("--profile_output", type=str, default=None, help="Output file for profile results (default: print to stdout)")

args = parser.parse_args()
# report config
logging.info(f"LBM root: {args.lbm_root}")
logging.info(f"Output root: {args.output_root}")
logging.info(f"Repo ID: {args.repo_id}")
logging.info(f"FPS: {args.fps}")
logging.info(f"Use videos: {args.use_videos}")
if args.keep_cameras is not None:
    logging.info(f"Keep cameras: {args.keep_cameras.split(',')}")
if args.state_keys is not None:
    logging.info(f"State keys: {args.state_keys.split(',')}")
if args.skill_types is not None:
    logging.info(f"Skill types: {args.skill_types.split(',')}")
if args.episode_dir_list is not None:
    logging.info(f"Episode directory list: {args.episode_dir_list}")
print("--------------------------------")

converter = LBMDatasetConverter(
    lbm_root=args.lbm_root,
    output_root=args.output_root,
    repo_id=args.repo_id,
    fps=args.fps,
    use_videos=args.use_videos,
    keep_cameras=args.keep_cameras.split(',') if args.keep_cameras is not None else None,
    state_keys=args.state_keys.split(',') if args.state_keys is not None else None,
    skill_types=args.skill_types.split(',') if args.skill_types is not None else None,
    episode_dir_list=args.episode_dir_list,
)

with profile_context(args.profile, args.profile_output):
    ds = converter.convert()
print(ds)

