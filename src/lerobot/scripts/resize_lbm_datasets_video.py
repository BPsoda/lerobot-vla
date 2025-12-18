#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import shutil
import sys
from typing import Tuple
import argparse
from functools import partial
import subprocess

import av
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF
from tqdm import tqdm

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_DATASET_ROOT = Path("/nfs_gaoyang/LBM_lerobot_dataset/LBM_lerobot_skills_dataset")
DEFAULT_TARGET_SIZE = (480, 640, 3)
DEFAULT_SUFFIX = "_resized"

# ==============================================================================
# Fast Video Resizing Core
# ==============================================================================

def resize_video_file(src_path: Path, dst_path: Path, target_shape: Tuple[int, int, int], device: str = "cpu"):
    """
    Directly resize an MP4 file using PyAV and Torch for maximum speed.
    """
    tH, tW, _ = target_shape
    
    try:
        container = av.open(str(src_path))
        stream = container.streams.video[0]
        
        out_container = av.open(str(dst_path), mode='w')
        # Use libsvtav1 to match the dataset's original AV1 format
        # preset 10 provides a good balance between speed and compression
        out_stream = out_container.add_stream('libsvtav1', rate=stream.average_rate)
        out_stream.width = tW
        out_stream.height = tH
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'preset': '10', 'crf': '30'}

        src_W, src_H = stream.width, stream.height
        scale = min(tW / src_W, tH / src_H)
        new_W, new_H = int(src_W * scale), int(src_H * scale)
        pad_left = (tW - new_W) // 2
        pad_right = tW - new_W - pad_left
        pad_top = (tH - new_H) // 2
        pad_bottom = tH - new_H - pad_top

        frames_batch = []
        
        def flush_batch(batch):
            if not batch: return
            imgs_np = np.stack([f.to_ndarray(format='rgb24') for f in batch])
            
            with torch.no_grad():
                imgs_torch = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(device)
                if device == "cuda": imgs_torch = imgs_torch.half()
                
                # High quality resize
                resized = TF.resize(imgs_torch, (new_H, new_W), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                # Robust pad
                padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                
                res_np = padded.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
                
                for i in range(len(batch)):
                    new_frame = av.VideoFrame.from_ndarray(res_np[i], format='rgb24')
                    new_frame.pts = batch[i].pts
                    new_frame.time_base = batch[i].time_base
                    for packet in out_stream.encode(new_frame):
                        out_container.mux(packet)

        for frame in container.decode(video=0):
            frames_batch.append(frame)
            if len(frames_batch) >= 16:
                flush_batch(frames_batch)
                frames_batch = []
                
        flush_batch(frames_batch)
        for packet in out_stream.encode():
            out_container.mux(packet)
            
        out_container.close()
        container.close()
    except Exception as e:
        logging.error(f"Failed to process video {src_path}: {e}")
        if dst_path.exists(): os.remove(dst_path)

def process_dataset(repo_id: str, dataset_root: Path, target_size: Tuple[int, int, int], suffix: str, device: str = "cpu"):
    try:
        dataset_path = dataset_root / repo_id
        new_repo_id = f"{repo_id}{suffix}"
        new_dataset_path = dataset_root / new_repo_id
        
        if new_dataset_path.exists():
            logging.info(f"[{repo_id}] Target dataset {new_repo_id} already exists. Skipping.")
            return

        # 1. Quick check info.json
        with open(dataset_path / "meta" / "info.json", "r") as f:
            info = json.load(f)
        
        needs_resize = False
        for key, feat in info["features"].items():
            if feat["dtype"] in ["image", "video"]:
                if tuple(feat["shape"]) != target_size:
                    needs_resize = True
                    break
        
        if not needs_resize:
            logging.info(f"[{repo_id}] Already matches target size. Skipping.")
            return

        logging.info(f"[{repo_id}] FAST VIDEO MODE: Resizing MP4 files to {target_size}...")

        # 2. Copy structure (data and meta)
        new_dataset_path.mkdir(parents=True, exist_ok=False)
        shutil.copytree(dataset_path / "data", new_dataset_path / "data", dirs_exist_ok=True)
        shutil.copytree(dataset_path / "meta", new_dataset_path / "meta", dirs_exist_ok=True)
        
        # 3. Update info.json
        info_path = new_dataset_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            info = json.load(f)
        for key, feat in info["features"].items():
            if feat["dtype"] in ["image", "video"]:
                feat["shape"] = list(target_size)
                # Update nested video info if present
                if "info" in feat:
                    if "video.height" in feat["info"]:
                        feat["info"]["video.height"] = target_size[0]
                    if "video.width" in feat["info"]:
                        feat["info"]["video.width"] = target_size[1]
                    if "video.codec" in feat["info"]:
                        feat["info"]["video.codec"] = "av1"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

        # 4. Process all video files
        video_files = list(dataset_path.glob("videos/**/*.mp4"))
        for src_mp4 in video_files:
            dst_mp4 = new_dataset_path / src_mp4.relative_to(dataset_path)
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            resize_video_file(src_mp4, dst_mp4, target_size, device)

        logging.info(f"[{repo_id}] FAST VIDEO MODE complete. Saved to {new_repo_id}")

    except Exception as e:
        logging.exception(f"Error processing dataset {repo_id}: {e}")

# ==============================================================================
# Helper for Multiprocessing Subprocesses
# ==============================================================================

def run_single_repo_cmd(repo_id, dataset_root, target_size, device, suffix):
    """
    Runs this script as a subprocess to process a single repo.
    """
    cmd = [
        sys.executable, __file__,
        "--repo-id", repo_id,
        "--dataset-root", str(dataset_root),
        "--target-size", *map(str, target_size),
        "--device", device,
        "--suffix", suffix
    ]
    # Use same environment to preserve PYTHONPATH
    subprocess.run(cmd, env=os.environ.copy(), check=True)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="FAST MP4 Direct Resizing for LBM Datasets")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--target-size", type=int, nargs=3, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--suffix", type=str, default=DEFAULT_SUFFIX)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repo-id", type=str, default=None, help="Process one repo only")
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
        # Worker mode
        process_dataset(args.repo_id, dataset_root, target_size, suffix, device=device)
    else:
        # Master mode
        dataset_dirs = [d.name for d in dataset_root.iterdir() 
                        if d.is_dir() and not d.name.endswith(suffix) 
                        and d.name.endswith("_ood_skill")
                        and (d / "meta" / "info.json").exists()]
        
        logging.info(f"Found {len(dataset_dirs)} potential datasets in {dataset_root}")
        
        num_workers = min(len(dataset_dirs), 8) # Fast mode is IO bound, too many workers might slow down disk
        logging.info(f"Using {num_workers} workers to spawn subprocesses.")
        
        worker_func = partial(
            run_single_repo_cmd, 
            dataset_root=dataset_root, 
            target_size=target_size, 
            device=device, 
            suffix=suffix
        )

        with Pool(processes=num_workers) as pool:
            pool.map(worker_func, sorted(dataset_dirs))

    logging.info("Fast resizing process completed.")

if __name__ == "__main__":
    main()

