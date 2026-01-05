#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool
import subprocess
import numpy as np
from tqdm import tqdm
import shutil
import sys
from typing import Tuple
import argparse
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

# Assume lerobot is in the PYTHONPATH
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# Configuration
# ==============================================================================

# Default values
DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")
DEFAULT_TARGET_SIZE = (480, 640, 3)
DEFAULT_SUFFIX = "_resized"

# Reserved keys managed by LeRobotDataset internally, should not be in the frame dict
RESERVED_KEYS = {"task_index", "frame_index", "episode_index", "timestamp", "index"}

# ==============================================================================
# Image Transformation Logic
# ==============================================================================

def resize_and_pad_torch(images: np.ndarray, target_shape: Tuple[int, int, int], device: str = "cpu", max_batch_size: int = 16) -> np.ndarray:
    """
    Resize a batch of images to fit target_shape while maintaining aspect ratio,
    then pad to target_shape using Torch for vectorized acceleration.
    Processes in chunks to avoid CUDA Out of Memory for long episodes.
    
    Args:
        images: np.ndarray of shape (N, H, W, C)
        target_shape: (target_H, target_W, target_C)
        device: 'cpu' or 'cuda'
        max_batch_size: Maximum frames to process at once on GPU
        
    Returns:
        np.ndarray of shape (N, target_H, target_W, target_C)
    """
    # 1. Detect input shape format: (N, H, W, C) or (N, C, H, W)
    # LeRobotDataset with torch transform returns (C, H, W)
    if images.shape[1] == 3 and images.ndim == 4:
        N, C, H, W = images.shape
        is_channels_first = True
    else:
        N, H, W, C = images.shape
        is_channels_first = False

    tH, tW, tC = target_shape
    
    # Pre-allocate output array on CPU (always H, W, C for LeRobot storage)
    output = np.empty((N, tH, tW, tC), dtype=np.uint8)
    
    # 2. Calculate scaling and padding
    scale = min(tW / W, tH / H)
    new_W, new_H = int(W * scale), int(H * scale)
    pad_left = (tW - new_W) // 2
    pad_right = tW - new_W - pad_left
    pad_top = (tH - new_H) // 2
    pad_bottom = tH - new_H - pad_top
    
    # Process in batches
    for i in range(0, N, max_batch_size):
        batch_end = min(i + max_batch_size, N)
        img_batch = images[i:batch_end]
        
        with torch.no_grad():
            imgs_torch = torch.from_numpy(img_batch).to(device)
            logging.debug(f"Imgs torch shape: {imgs_torch.shape}")
            if not is_channels_first:
                imgs_torch = imgs_torch.permute(0, 3, 1, 2)
            
            if device == "cuda":
                imgs_torch = imgs_torch.half()
            
            # 3. Vectorized Interpolation using torchvision.v2
            # InterpolationMode.BILINEAR with antialias=True is high quality.
            resized = TF.resize(imgs_torch, (new_H, new_W), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            logging.debug(f"Resized shape: {resized.shape}")

            # 4. Vectorized Padding
            # Using F.pad (torch.nn.functional.pad) for 4D tensors is more robust.
            # F.pad padding order is (padding_left, padding_right, padding_top, padding_bottom)
            padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            logging.debug(f"Padded shape: {padded.shape}")
            
            # 5. Permute back to (N, H, W, C) for storage
            final_batch = padded.permute(0, 2, 3, 1)
            if device == "cuda":
                final_batch = final_batch.clamp(0, 255).to(torch.uint8)
            
            output[i:batch_end] = final_batch.cpu().numpy()
            logging.debug(f"Output shape: {output.shape}")

            # Optional: Clean up to free memory faster
            del imgs_torch, resized, padded, final_batch
            if device == "cuda" and i % (max_batch_size * 10) == 0:
                torch.cuda.empty_cache()

    return output

# ==============================================================================
# Helper for Multiprocessing Subprocesses
# ==============================================================================

def run_single_repo_cmd(repo_id, dataset_root, target_size, device, suffix):
    """
    Function to run the script itself as a subprocess for a single repo.
    Must be at top-level to be picklable by multiprocessing.
    """
    cmd = [
        sys.executable, __file__,
        "--repo-id", repo_id,
        "--dataset-root", str(dataset_root),
        "--target-size", *map(str, target_size),
        "--device", device,
        "--suffix", suffix
    ]
    logging.info(f"Spawning subprocess for {repo_id}...")
    # Use same environment to preserve PYTHONPATH
    # TODO: if the subprocess fails, record the repo_id to a file
    result = subprocess.run(cmd, env=os.environ.copy(), check=True)
    if result.returncode != 0:
        logging.error(f"Subprocess for {repo_id} failed: {result.stderr}")
        with open("failed_resize_lbm_eval_skills_repo_ids.txt", "a") as f:
            f.write(f"{repo_id}\n")
    else:
        logging.info(f"Subprocess for {repo_id} succeeded")

# ==============================================================================
# Worker Function
# ==============================================================================

def process_dataset(repo_id: str, dataset_root: Path, target_size: Tuple[int, int, int], suffix: str, device: str = "cpu"):
    try:
        dataset_path = dataset_root / repo_id
        if not dataset_path.exists() or not (dataset_path / "meta" / "info.json").exists():
            return

        with open(dataset_path / "meta" / "info.json", "r") as f:
            info = json.load(f)
        
        needs_resize = False
        image_keys = []
        no_resize_image_keys = []
        for key, feat in info["features"].items():
            if feat["dtype"] in ["image", "video"]:
                current_shape = tuple(feat["shape"])
                if current_shape != target_size:
                    needs_resize = True
                    image_keys.append(key)
                else:
                    no_resize_image_keys.append(key)
        logging.debug(f"[{repo_id}] Image keys: {image_keys}")
        
        if not needs_resize:
            logging.info(f"[{repo_id}] Already matches target size {target_size}. Skipping.")
            return

        new_repo_id = f"{repo_id}{suffix}"
        new_dataset_path = dataset_root / new_repo_id
        
        if new_dataset_path.exists():
            logging.info(f"[{repo_id}] Target dataset {new_repo_id} already exists. Skipping.")
            return

        logging.info(f"[{repo_id}] Dimensions mismatch. Resizing to {target_size} using {device}...")

        src_ds = LeRobotDataset(repo_id, root=dataset_path, video_backend="torchcodec")
        new_features = src_ds.meta.features.copy()
        for key in image_keys:
            new_features[key] = new_features[key].copy()
            new_features[key]["shape"] = list(target_size)
        
        dst_ds = LeRobotDataset.create(
            repo_id=new_repo_id,
            fps=src_ds.fps,
            features=new_features,
            root=new_dataset_path,
            use_videos=len(src_ds.meta.video_keys) > 0,
            video_backend="torchcodec",
        )

        for ep_idx in tqdm(range(src_ds.num_episodes), desc=f"Resizing {repo_id}"):
            episode_data = src_ds.hf_dataset.select(range(
                src_ds.meta.episodes["dataset_from_index"][ep_idx],
                src_ds.meta.episodes["dataset_to_index"][ep_idx]
            ))
            
            # 确定 episode 的起止索引
            start_idx = src_ds.meta.episodes["dataset_from_index"][ep_idx]
            end_idx = src_ds.meta.episodes["dataset_to_index"][ep_idx]
            num_frames = end_idx - start_idx
            
            # 从 src_ds 中获取完整的 frame 列表（包含图像和低维数据）
            # LeRobotDataset[i] 会自动处理视频解码
            frames = [src_ds[i] for i in range(start_idx, end_idx)]
            
            # 将列表转换为字典，方便批量处理
            feature_keys = [k for k in src_ds.features if k not in RESERVED_KEYS]
            batch = {key: [] for key in feature_keys}
            has_task = "task" in frames[0]
            if has_task:
                batch["task"] = []
                
            for f in frames:
                for key in feature_keys:
                    if key in RESERVED_KEYS:
                        continue
                    val = f[key]
                    # LeRobotDataset 默认可能返回 Tensor，转回 numpy
                    batch[key].append(val.cpu().numpy() if torch.is_tensor(val) else np.array(val))
                if has_task:
                    batch["task"].append(f["task"])
            
            processed_episode = {}
            for key in feature_keys:
                if key not in image_keys and key not in no_resize_image_keys:
                    processed_episode[key] = np.array(batch[key])
                elif key in no_resize_image_keys:
                    processed_episode[key] = np.array(batch[key]).transpose(0, 2, 3, 1)
                else:
                    # 批量转换图像 (N, H, W, C)
                    raw_images = np.array(batch[key])
                    processed_episode[key] = resize_and_pad_torch(raw_images, target_size, device=device)
            
            for i in range(num_frames):
                frame = {key: processed_episode[key][i] for key in feature_keys}
                if has_task:
                    frame["task"] = batch["task"][i]
                dst_ds.add_frame(frame)
            
            dst_ds.save_episode()

        dst_ds._close_writer()
        dst_ds.meta._close_writer()
        
        if (dataset_path / "meta" / "stats.json").exists():
            shutil.copy(dataset_path / "meta" / "stats.json", new_dataset_path / "meta" / "stats.json")
        
        logging.info(f"[{repo_id}] Successfully saved to {new_repo_id}")

    except Exception as e:
        logging.exception(f"Error processing dataset {repo_id}: {e}")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Resize LBM datasets to target size using Torch acceleration")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Root directory containing datasets")
    parser.add_argument("--target-size", type=int, nargs=3, default=DEFAULT_TARGET_SIZE, help="Target size (H W C)")
    parser.add_argument("--suffix", type=str, default=DEFAULT_SUFFIX, help="Suffix for new datasets")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for resizing")
    parser.add_argument("--repo-id", type=str, default=None, help="Process a single specific repo-id")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout
    )

    dataset_root = Path(args.dataset_root)
    target_size = tuple(args.target_size)
    suffix = args.suffix
    device = args.device

    if args.repo_id:
        # Single task mode: process one dataset
        process_dataset(args.repo_id, dataset_root, target_size, suffix, device=device)
    else:
        # Batch mode: spawn subprocesses for each dataset
        dataset_dirs = [d.name for d in dataset_root.iterdir() 
                        if d.is_dir() and not d.name.endswith(suffix) 
                        and (d.name.endswith("_train") or d.name.endswith("_iid") or d.name.endswith("_ood_manipuland") or d.name.endswith("_ood_operator"))
                        and (d / "meta" / "info.json").exists()]
        
        logging.info(f"Found {len(dataset_dirs)} potential datasets in {dataset_root}")
        
        num_workers = 16
        logging.info(f"Using {num_workers} workers to spawn subprocesses.")
        
        # Using partial to pass fixed arguments to the global function
        worker_func = partial(
            run_single_repo_cmd, 
            dataset_root=dataset_root, 
            target_size=target_size, 
            device=device, 
            suffix=suffix
        )

        with Pool(processes=num_workers) as pool:
            pool.map(worker_func, sorted(dataset_dirs))

    logging.info("Resizing process completed.")

if __name__ == "__main__":
    main()
