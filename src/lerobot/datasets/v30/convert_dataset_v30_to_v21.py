#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script will help you convert any LeRobot dataset from codebase version 3.0 to 2.1. It will:

- Convert episodes metadata from parquet to `episodes.jsonl`
- Convert episodes stats from parquet to `episodes_stats.jsonl`
- Convert tasks from parquet to `tasks.jsonl`
- Split concatenated data files back into individual episode files
- Split concatenated video files back into individual episode videos
- Update codebase_version in `info.json` and restore v2.1-specific fields
- Optionally push this new version to the hub on the 'main' branch and tag it with "v2.1".

Usage:

Convert a dataset from the hub:
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht
```

Convert a local dataset (works in place):
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/local/dataset/directory \
    --push-to-hub=false
```

Convert a local dataset to a different output directory:
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/input/dataset/directory \
    --output-root=/path/to/output/dataset/directory \
    --push-to-hub=false
```

"""

import argparse
import logging
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Features, Image
from huggingface_hub import HfApi, snapshot_download
from requests import HTTPError

from lerobot.datasets.compute_stats import compute_episode_stats, get_feature_stats, DEFAULT_QUANTILES
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_episodes,
    load_info,
    load_nested_dataset,
    load_tasks,
    write_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

"""
-------------------------
NEW (v3.0)
data/chunk-000/file_000.parquet

OLD (v2.1)
data/chunk-000/episode_000000.parquet
-------------------------
NEW (v3.0)
videos/CAMERA/chunk-000/file_000.mp4

OLD (v2.1)
videos/chunk-000/CAMERA/episode_000000.mp4
-------------------------
NEW (v3.0)
meta/episodes/chunk-000/episodes_000.parquet
episode_index | video_chunk_index | video_file_index | data/chunk_index | data/file_index | tasks | length

OLD (v2.1)
episodes.jsonl
{"episode_index": 1, "tasks": ["Put the blue block in the green bowl"], "length": 266}
-------------------------
NEW (v3.0)
meta/tasks.parquet
task_index | task

OLD (v2.1)
tasks.jsonl
{"task_index": 1, "task": "Put the blue block in the green bowl"}
-------------------------
NEW (v3.0)
meta/episodes_stats/chunk-000/file_000.parquet
episode_index | stats/... (flattened)

OLD (v2.1)
episodes_stats.jsonl
{"episode_index": 1, "stats": {...}}
-------------------------
UPDATE
meta/info.json
-------------------------
"""


def write_jsonlines(data: list[dict], fpath: Path) -> None:
    """Write data to a JSONL file."""
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, "w") as writer:
        for item in data:
            writer.write(item)


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v3.0 version."""
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script is specifically for converting v3.0 datasets to v2.1."
        )


def convert_tasks(root: Path, new_root: Path, overwrite: bool = False) -> None:
    """Convert tasks from v3.0 parquet format to v2.1 jsonl format."""
    tasks_path = new_root / LEGACY_TASKS_PATH
    if tasks_path.exists() and not overwrite:
        logging.info(f"Tasks file already exists at {tasks_path}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting tasks from {root} to {new_root}")
    tasks_df = load_tasks(root)
    tasks_list = []
    # In v3.0, task strings are stored as the DataFrame index, not as a column
    for task_string, row in tasks_df.iterrows():
        tasks_list.append({"task_index": int(row["task_index"]), "task": task_string})
    tasks_list = sorted(tasks_list, key=lambda x: x["task_index"])
    write_jsonlines(tasks_list, tasks_path)


def convert_episodes(root: Path, new_root: Path, overwrite: bool = False) -> None:
    """Convert episodes metadata from v3.0 parquet format to v2.1 jsonl format."""
    episodes_path = new_root / LEGACY_EPISODES_PATH
    if episodes_path.exists() and not overwrite:
        logging.info(f"Episodes file already exists at {episodes_path}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting episodes metadata from {root} to {new_root}")
    episodes_ds = load_episodes(root)
    episodes_list = []
    for episode in episodes_ds:
        ep_dict = {
            "episode_index": int(episode["episode_index"]),
            "tasks": episode.get("tasks", []),
            "length": int(episode["length"]),
        }
        episodes_list.append(ep_dict)
    episodes_list = sorted(episodes_list, key=lambda x: x["episode_index"])
    write_jsonlines(episodes_list, episodes_path)


def convert_episodes_stats(root: Path, new_root: Path, repo_id: str | None = None, overwrite: bool = False) -> None:
    """Convert episodes stats from v3.0 parquet format to v2.1 jsonl format.
    
    If stats don't exist in the episodes metadata, they will be computed from the episode data.
    
    Args:
        root: Root directory of the v3.0 dataset
        new_root: Root directory where the v2.1 dataset will be written
        repo_id: Repository ID for loading the dataset
        overwrite: If False and episodes_stats.jsonl already exists, skip computation
    """
    episodes_stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    if episodes_stats_path.exists() and not overwrite:
        logging.info(f"Episodes stats already exist at {episodes_stats_path}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting episodes stats from {root} to {new_root}")
    episodes_ds = load_episodes(root)
    episodes_stats_list = []
    has_existing_stats = False
    
    # First, try to extract stats from episodes metadata
    for episode in episodes_ds:
        # Extract stats from flattened keys (stats/...)
        stats_dict = {}
        has_stats = False
        for key, value in episode.items():
            if key.startswith("stats/"):
                has_stats = True
                has_existing_stats = True
                # Convert stats/key/subkey to nested dict
                stats_key = key.replace("stats/", "")
                parts = stats_key.split("/")
                current = stats_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    current[parts[-1]] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    current[parts[-1]] = value.item()
                else:
                    current[parts[-1]] = value

        if has_stats:
            ep_stats_dict = {
                "episode_index": int(episode["episode_index"]),
                "stats": stats_dict,
            }
            episodes_stats_list.append(ep_stats_dict)
    
    # If no stats found in metadata, compute them from episode data
    if not has_existing_stats:
        logging.info("No episodes stats found in metadata. Computing stats from episode data...")
        info = load_info(root)
        features = info["features"]
        
        # Load the dataset to access episode data
        # root is the full dataset directory (e.g., /path/to/dataset/repo_id)
        # So we use root.parent as the base directory and repo_id (or root.name) as the dataset name
        if repo_id is None:
            repo_id = root.name
        
        dataset = LeRobotDataset(repo_id=repo_id, root=root.parent, download_videos=False)
        
        for episode in tqdm.tqdm(episodes_ds, desc="Computing episode stats"):
            ep_idx = int(episode["episode_index"])
            from_index = int(episode["dataset_from_index"])
            to_index = int(episode["dataset_to_index"])
            
            # Collect episode data - separate numerical and image/video features
            numerical_data = {}
            image_data = {}
            
            for idx in range(from_index, to_index):
                item = dataset[idx]
                
                for key, value in item.items():
                    if key not in features:
                        continue
                    if key in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
                        continue
                    
                    # Handle image/video features: collect arrays directly
                    if features[key]["dtype"] in ["image", "video"]:
                        if key not in image_data:
                            image_data[key] = []
                        
                        # Convert torch tensors to numpy
                        if hasattr(value, "cpu"):
                            img_array = value.cpu().numpy()
                        elif hasattr(value, "numpy"):
                            img_array = value.numpy()
                        else:
                            img_array = np.array(value)
                        
                        # Handle different array formats: (C, H, W) or (H, W, C)
                        if img_array.ndim == 3:
                            if img_array.shape[0] == 3 or img_array.shape[0] == 1:
                                # Channel first: (C, H, W) -> (H, W, C)
                                img_array = np.transpose(img_array, (1, 2, 0))
                            # Now it's (H, W, C)
                            
                            # Convert to uint8 if needed (for consistent stats computation)
                            if img_array.dtype != np.uint8:
                                if img_array.max() <= 1.0:
                                    img_array = (img_array * 255).astype(np.uint8)
                                else:
                                    img_array = img_array.astype(np.uint8)
                            
                            image_data[key].append(img_array)
                    else:
                        # Numerical features
                        if key not in numerical_data:
                            numerical_data[key] = []
                        
                        # Convert torch tensors to numpy
                        if hasattr(value, "cpu"):
                            value = value.cpu().numpy()
                        elif hasattr(value, "numpy"):
                            value = value.numpy()
                        
                        numerical_data[key].append(value)
            
            # Stack arrays
            for key in numerical_data:
                numerical_data[key] = np.stack(numerical_data[key])
            
            for key in image_data:
                # Stack images: (N, H, W, C) format
                image_data[key] = np.stack(image_data[key])
            
            # Compute stats for numerical features using compute_episode_stats
            # (only pass numerical features to avoid image I/O)
            # Filter to only include keys that exist in features dict and have valid dtype
            numerical_data_filtered = {}
            for k, v in numerical_data.items():
                if k in features and features[k].get("dtype") not in ["image", "video", "string"]:
                    numerical_data_filtered[k] = v
            
            ep_stats = {}
            if numerical_data_filtered:
                try:
                    # Create a filtered features dict that only contains keys in numerical_data_filtered
                    filtered_features = {k: v for k, v in features.items() if k in numerical_data_filtered}
                    ep_stats = compute_episode_stats(numerical_data_filtered, filtered_features)
                except Exception as e:
                    # Fallback: compute stats manually for numerical features
                    logging.warning(f"Error computing stats for episode {ep_idx}: {e}. Computing basic stats only.")
                    for key, data in numerical_data_filtered.items():
                        if key not in features:
                            continue
                        if features[key].get("dtype") == "string":
                            continue
                        axes_to_reduce = 0
                        keepdims = data.ndim == 1
                        ep_stats[key] = get_feature_stats(
                            data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
                        )
            
            # Compute stats directly for image/video features from arrays
            # Filter to only include keys that exist in features dict
            for key, img_array in image_data.items():
                if key not in features:
                    continue
                
                # Downsample images for faster stats computation (stats don't need full resolution)
                # Target size: 64x64 is sufficient for accurate statistics
                target_size = 64
                N, H, W, C = img_array.shape
                if H > target_size or W > target_size:
                    # Downsample using area interpolation (good for downsampling)
                    downsampled = np.zeros((N, target_size, target_size, C), dtype=img_array.dtype)
                    for i in range(N):
                        downsampled[i] = cv2.resize(
                            img_array[i], 
                            (target_size, target_size), 
                            interpolation=cv2.INTER_NEAREST
                        )
                    img_array = downsampled
                
                # img_array is (N, H, W, C) - transpose to (N, C, H, W) to match expected format
                # This matches the format expected by get_feature_stats with axis=(0, 2, 3)
                img_array = np.transpose(img_array, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
                
                # Compute per-channel stats by reducing over (batch, height, width)
                axes_to_reduce = (0, 2, 3)  # Reduce over batch, height, width (keep channels)
                keepdims = True
                
                img_stats = get_feature_stats(
                    img_array, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
                )
                
                # Normalize image stats to [0,1] (divide by 255.0)
                img_stats = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in img_stats.items()
                }
                
                ep_stats[key] = img_stats
            
            # Convert numpy arrays to lists for JSON serialization
            stats_dict_serialized = {}
            for key, value in ep_stats.items():
                if isinstance(value, dict):
                    stats_dict_serialized[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                        for k, v in value.items()
                    }
                else:
                    stats_dict_serialized[key] = value.tolist() if isinstance(value, np.ndarray) else value
            
            ep_stats_dict = {
                "episode_index": ep_idx,
                "stats": stats_dict_serialized,
            }
            episodes_stats_list.append(ep_stats_dict)
    
    if episodes_stats_list:
        episodes_stats_list = sorted(episodes_stats_list, key=lambda x: x["episode_index"])
        write_jsonlines(episodes_stats_list, new_root / LEGACY_EPISODES_STATS_PATH)
        logging.info(f"Successfully converted/computed stats for {len(episodes_stats_list)} episodes")
    else:
        logging.warning("No episodes stats found or computed. Skipping episodes_stats.jsonl creation.")


def _get_encoder_codec(decoder_codec_name: str) -> str:
    """Map decoder codec names to encoder codec names."""
    # Map common decoder codecs to encoder codecs
    codec_mapping = {
        "libdav1d": "libsvtav1",  # AV1 decoder -> AV1 encoder
        "av1": "libsvtav1",
        "hevc": "hevc",  # H.265
        "h265": "hevc",
        "h264": "h264",
        "libx264": "h264",
        "libx265": "hevc",
    }
    
    # Try to find a match (case-insensitive)
    decoder_lower = decoder_codec_name.lower()
    for decoder, encoder in codec_mapping.items():
        if decoder in decoder_lower:
            return encoder
    
    # Default to h264 if no match found (widely supported)
    return "h264"


def split_video_file(
    video_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> None:
    """Extract a segment from a video file using pyav."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with av.open(str(video_path)) as input_container:
        if not input_container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")
        
        video_stream = input_container.streams.video[0]
        fps = float(video_stream.average_rate)
        decoder_codec = video_stream.codec.name
        encoder_codec = _get_encoder_codec(decoder_codec)
        
        output_container = av.open(str(output_path), mode="w")
        fps_fraction = Fraction(fps).limit_denominator(1000)
        output_stream = output_container.add_stream(encoder_codec, rate=fps_fraction)
        
        # Copy stream properties from input
        output_stream.width = video_stream.codec_context.width
        output_stream.height = video_stream.codec_context.height
        output_stream.pix_fmt = video_stream.codec_context.pix_fmt
        output_stream.time_base = Fraction(1, int(fps))
        
        # Seek to start time
        input_container.seek(int(start_time * av.time_base))
        
        end_time = start_time + duration
        frame_count = 0
        
        # Decode and encode frames
        for packet in input_container.demux(video_stream):
            for frame in packet.decode():
                if frame is None:
                    continue
                
                frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                
                if frame_time >= end_time:
                    break
                
                if frame_time >= start_time:
                    # Create new frame with reset timestamps
                    new_frame = frame.reformat(
                        width=output_stream.width,
                        height=output_stream.height,
                        format=output_stream.pix_fmt
                    )
                    new_frame.pts = frame_count
                    new_frame.time_base = Fraction(1, int(fps))
                    
                    for packet_out in output_stream.encode(new_frame):
                        output_container.mux(packet_out)
                    
                    frame_count += 1
            else:
                continue
            break
        
        # Flush encoder
        for packet_out in output_stream.encode():
            output_container.mux(packet_out)
        
        output_container.close()


def convert_videos(root: Path, new_root: Path, overwrite: bool = False) -> None:
    """Convert videos from v3.0 format to v2.1 format."""
    videos_dir = new_root / "videos"
    if videos_dir.exists() and any(videos_dir.glob("*/*/*.mp4")) and not overwrite:
        logging.info(f"Video files already exist in {videos_dir}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting videos from {root} to {new_root}")

    info = load_info(root)
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    if len(video_keys) == 0:
        return

    episodes_ds = load_episodes(root)
    video_keys = sorted(video_keys)

    # Group episodes by video file
    video_file_map = {}  # (video_key, chunk_idx, file_idx) -> list of episodes
    for episode in episodes_ds:
        for video_key in video_keys:
            chunk_idx = episode.get(f"videos/{video_key}/chunk_index")
            file_idx = episode.get(f"videos/{video_key}/file_index")
            if chunk_idx is not None and file_idx is not None:
                key = (video_key, int(chunk_idx), int(file_idx))
                if key not in video_file_map:
                    video_file_map[key] = []
                video_file_map[key].append(episode)

    # Process each video file
    for (video_key, chunk_idx, file_idx), episodes in tqdm.tqdm(
        video_file_map.items(), desc="Converting videos"
    ):
        video_path = root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            continue

        # Sort episodes by episode_index
        episodes = sorted(episodes, key=lambda x: x["episode_index"])

        for episode in episodes:
            ep_idx = int(episode["episode_index"])
            from_timestamp = float(episode.get(f"videos/{video_key}/from_timestamp", 0.0))
            to_timestamp = float(episode.get(f"videos/{video_key}/to_timestamp", 0.0))
            duration = to_timestamp - from_timestamp

            # Determine chunk for output (v2.1 format)
            output_chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
            output_path = new_root / "videos" / f"chunk-{output_chunk_idx:03d}" / video_key / f"episode_{ep_idx:06d}.mp4"
            
            # Skip if file exists and overwrite is False
            if output_path.exists() and not overwrite:
                continue

            split_video_file(video_path, output_path, from_timestamp, duration)


def convert_data(root: Path, new_root: Path, overwrite: bool = False) -> None:
    """Convert data files from v3.0 format to v2.1 format."""
    data_dir = new_root / "data"
    if data_dir.exists() and any(data_dir.glob("*/*.parquet")) and not overwrite:
        logging.info(f"Data files already exist in {data_dir}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting data files from {root} to {new_root}")

    episodes_ds = load_episodes(root)
    info = load_info(root)
    features = info["features"]
    image_keys = [key for key, ft in features.items() if ft["dtype"] == "image"]

    # Load all data files
    data_dir = root / "data"
    data_files = sorted(data_dir.glob("*/*.parquet"))
    if len(data_files) == 0:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    # Load all data into memory (for simplicity, could be optimized for large datasets)
    all_data = load_nested_dataset(data_dir, features=None)

    # Process each episode
    for episode in tqdm.tqdm(episodes_ds, desc="Converting data files"):
        ep_idx = int(episode["episode_index"])
        from_index = int(episode["dataset_from_index"])
        to_index = int(episode["dataset_to_index"])

        # Extract episode data
        episode_data = all_data[from_index:to_index]

        # Convert to DataFrame
        episode_df = pd.DataFrame(episode_data)

        # Determine chunk for output (v2.1 format)
        output_chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
        output_path = new_root / "data" / f"chunk-{output_chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        
        # Skip if file exists and overwrite is False
        if output_path.exists() and not overwrite:
            continue
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write parquet file
        if len(image_keys) > 0:
            schema = pa.Schema.from_pandas(episode_df)
            hf_features = Features.from_arrow_schema(schema)
            for key in image_keys:
                if key in hf_features:
                    hf_features[key] = Image()
            schema = hf_features.arrow_schema
            episode_df.to_parquet(output_path, index=False, schema=schema)
        else:
            episode_df.to_parquet(output_path, index=False)


def convert_info(root: Path, new_root: Path, overwrite: bool = False) -> None:
    """Convert info.json from v3.0 to v2.1 format."""
    info_path = new_root / INFO_PATH
    if info_path.exists() and not overwrite:
        logging.info(f"Info file already exists at {info_path}. Skipping (use --overwrite to recompute).")
        return
    
    logging.info(f"Converting info from {root} to {new_root}")
    info = load_info(root)
    info["codebase_version"] = V21

    # Remove v3.0-specific fields
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    info.pop("data_path", None)
    info.pop("video_path", None)

    # Add v2.1-specific fields
    # Calculate total_chunks (based on number of episodes)
    total_episodes = info.get("total_episodes", 0)
    if total_episodes > 0:
        info["total_chunks"] = (total_episodes - 1) // DEFAULT_CHUNK_SIZE + 1
    else:
        info["total_chunks"] = 0

    # Calculate total_videos (number of episodes with videos)
    # In v2.1, each episode has its own video file, so total_videos = number of episodes with videos
    videos_dir = root / "videos"
    if videos_dir.exists():
        # Count episodes that have video metadata
        episodes_ds = load_episodes(root)
        video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
        episodes_with_videos = 0
        for episode in episodes_ds:
            has_video = False
            for video_key in video_keys:
                if episode.get(f"videos/{video_key}/chunk_index") is not None:
                    has_video = True
                    break
            if has_video:
                episodes_with_videos += 1
        info["total_videos"] = episodes_with_videos
    else:
        info["total_videos"] = 0

    # Remove fps from individual features (v2.1 has fps at top level only)
    for key in info["features"]:
        if "fps" in info["features"][key]:
            del info["features"][key]["fps"]

    write_info(info, new_root)


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    root: str | Path | None = None,
    output_root: str | Path | None = None,
    push_to_hub: bool = True,
    force_conversion: bool = False,
    overwrite: bool = False,
):
    """Convert a dataset from v3.0 to v2.1 format.
    
    Args:
        repo_id: Repository identifier on Hugging Face.
        branch: Repo branch to push your dataset. Defaults to the main branch.
        root: Local directory containing the input dataset (v3.0).
        output_root: Local directory where the converted dataset (v2.1) will be written.
                    If None, uses the same parent directory as root.
        push_to_hub: Whether to push the converted dataset to the hub.
        force_conversion: Force conversion even if the dataset already has a v2.1 version.
        overwrite: If False, skip conversion of files that already exist (e.g., episodes_stats.jsonl).
    """
    # First check if the dataset already has a v2.1 version
    if root is None and not force_conversion:
        try:
            print("Trying to download v2.1 version of the dataset from the hub...")
            snapshot_download(repo_id, repo_type="dataset", revision=V21, local_dir=HF_LEROBOT_HOME / repo_id)
            return
        except Exception:
            print("Dataset does not have an uploaded v2.1 version. Continuing with conversion.")

    # Set root based on whether local dataset path is provided
    use_local_dataset = False
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root) / repo_id
    if root.exists():
        validate_local_dataset_version(root)
        use_local_dataset = True
        print(f"Using local dataset at {root}")

    # Set output_root - if not specified, use same parent as root
    if output_root is None:
        old_root = root.parent / f"{root.name}_old"
        new_root = root.parent / f"{root.name}_v21"
    else:
        output_root = Path(output_root)
        old_root = root.parent / f"{root.name}_old"
        new_root = output_root / repo_id

    # Handle old_root cleanup if both old_root and root exist
    if old_root.is_dir() and root.is_dir() and overwrite:
        shutil.rmtree(str(root))
        shutil.move(str(old_root), str(root))

    if new_root.is_dir() and overwrite:
        shutil.rmtree(new_root)

    if not use_local_dataset:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=V30,
            local_dir=root,
        )

    convert_info(root, new_root, overwrite=overwrite)
    convert_tasks(root, new_root, overwrite=overwrite)
    convert_episodes(root, new_root, overwrite=overwrite)
    convert_episodes_stats(root, new_root, repo_id=repo_id, overwrite=overwrite)
    convert_data(root, new_root, overwrite=overwrite)
    convert_videos(root, new_root, overwrite=overwrite)

    # Only move/rename if output_root was not specified (in-place conversion)
    if output_root is None:
        shutil.move(str(root), str(old_root))
        shutil.move(str(new_root), str(root))
    else:
        # If output_root was specified, keep original dataset and just use new_root as final location
        print(f"Converted dataset saved to {new_root}")

    if push_to_hub:
        hub_api = HfApi()
        try:
            hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
        except HTTPError as e:
            print(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
            pass
        hub_api.delete_files(
            delete_patterns=["data/chunk*/file_*", "meta/episodes/chunk*", "meta/tasks.parquet", "meta/episodes_stats/chunk*", "videos/*/chunk*"],
            repo_id=repo_id,
            revision=branch,
            repo_type="dataset",
        )
        hub_api.create_tag(repo_id, tag=V21, revision=branch, repo_type="dataset")

        LeRobotDataset(repo_id).push_to_hub()


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory containing the input dataset (v3.0).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Local directory where the converted dataset (v2.1) will be written. "
        "If not specified, the converted dataset will be written to the same parent directory as the input.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda input: input.lower() == "true",
        default=True,
        help="Push the converted dataset to the hub.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force conversion even if the dataset already has a v2.1 version.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist. If False, skips conversion of files that already exist.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))

