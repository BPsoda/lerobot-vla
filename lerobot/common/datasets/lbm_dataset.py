"""
LBM diffusion-spartan -> LeRobotDataset converter.

Usage:
    from lerobot.datasets.lbm_dataset import LBMDatasetConverter

    converter = LBMDatasetConverter(
        lbm_root="/path/to/extracted/tasks",  # tasks/Skill/.../episode_*/processed
        output_root="/tmp/lerobot_datasets",
        repo_id="lbm-eval-v1",
        fps=10,
        use_videos=True,          # store RGB as MP4; set False to keep PNGs
        keep_cameras=None,        # or a set like {"front", "over_shoulder"}
    )
    ds = converter.convert()
    # Train with: train.py --dataset.repo_id lbm-eval-v1 --dataset.root /tmp/lerobot_datasets

Output dataset fields (LeRobot format):
    action                       float32[20]   # ordered per ACTION_NAMES below
    observation.images.<cam>     video/image   # HxWx3 RGB for each kept camera
    index, episode_index,
    frame_index, timestamp,
    task_index                   standard default fields from LeRobot

Assumptions:
    - LBM data extracted locally with layout documented in LBM_TRAINING_DATASET.md
    - Each processed episode has observations.npz (RGB, depth, labels) and actions.npz (20-dim)
    - Only RGB streams are ingested; depth/label ignored by design
"""

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from tqdm import tqdm
import re
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES

ACTION_NAMES = [
    "right_xyz_x",
    "right_xyz_y",
    "right_xyz_z",
    "right_rot6d_a",
    "right_rot6d_b",
    "right_rot6d_c",
    "right_rot6d_d",
    "right_rot6d_e",
    "right_rot6d_f",
    "left_xyz_x",
    "left_xyz_y",
    "left_xyz_z",
    "left_rot6d_a",
    "left_rot6d_b",
    "left_rot6d_c",
    "left_rot6d_d",
    "left_rot6d_e",
    "left_rot6d_f",
    "right_gripper",
    "left_gripper",
]


def _sanitize_name(raw: str) -> str:
    """Convert arbitrary camera names into a safe snake_case token."""
    safe = raw.replace(" ", "_").replace("-", "_")
    return safe.lower()


def pascal_to_snake_case(pascal_str):
    # 使用正则表达式匹配大写字母并在它们前面加上空格，最后转换为小写
    result = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', pascal_str).lower()
    return result

def resize_image(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Resize an image to a given shape."""
    image = Image.fromarray(image)
    # assuming the last dimension to be the color channel
    h, w, _ = shape
    image = image.resize((w, h))
    return np.array(image)


def resize_and_pad(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Resize a batch of images to fit target_shape while maintaining aspect ratio,
    then pad to target_shape.
    
    Args:
        image: Input image as numpy array of shape (H, W, C) or (H, W)
        target_shape: Target shape as (H, W, C) tuple
        
    Returns:
        Resized and padded image of shape target_shape
    """
    target_h, target_w, target_c = target_shape
    
    # Handle grayscale images
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    
    h, w, c = image.shape
    
    # Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image maintaining aspect ratio
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(pil_image)
    
    # Ensure correct number of channels
    if resized.ndim == 2:
        resized = np.expand_dims(resized, axis=2)
    if resized.shape[2] != target_c:
        if target_c == 3 and resized.shape[2] == 1:
            # Convert grayscale to RGB
            resized = np.repeat(resized, target_c, axis=2)
        elif target_c == 1 and resized.shape[2] == 3:
            # Convert RGB to grayscale
            resized = np.mean(resized, axis=2, keepdims=True)
        else:
            raise ValueError(f"Cannot convert {resized.shape[2]} channels to {target_c} channels")
    
    # Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    
    # Pad symmetrically (or pad bottom/right if odd)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding (using zero padding)
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    return padded.astype(image.dtype)

class LBMDatasetConverter:
    """
    Utility to convert the LBM training dataset (diffusion spartan format) into
    the LeRobotDataset format used by the training pipeline.

    The converter expects the LBM dataset to be extracted locally with the
    layout described in ``LBM_TRAINING_DATASET.md``:

    tasks/{SKILL}/{STATION}/sim/bc/teleop/{DATE}/diffusion_spartan/episode_*/processed

    Each processed directory must contain ``observations.npz`` and
    ``actions.npz``. The converter keeps:
    - RGB streams for each camera (depth/labels are ignored by default).
    - The 20-dim action vector per frame.

    Usage example:
        converter = LBMDatasetConverter(
            lbm_root="/path/to/extracted/tasks",
            output_root="/tmp/lerobot_datasets",
            repo_id="lbm-eval-v1",
        )
        lerobot_ds = converter.convert()
    """

    def __init__(
        self,
        lbm_root: str | Path,
        output_root: str | Path | None = None,
        repo_id: str = "lbm-eval",
        fps: int = 10,
        use_videos: bool = True,
        keep_cameras: Iterable[str] | None = None,
        state_keys: Iterable[str] | None = None,
        skill_types: Iterable[str] | None = None,
        episode_dir_list: str | Path | None = None,
    ) -> None:
        self.lbm_root = Path(lbm_root)
        self.output_root = Path(output_root) if output_root is not None else None
        self.repo_id = repo_id
        self.fps = fps
        self.use_videos = use_videos
        self.keep_cameras = list(keep_cameras) if keep_cameras is not None else None
        self.state_keys = list(state_keys) if state_keys is not None else None
        self.skill_types = list(skill_types) if skill_types is not None else None
        self.episode_dir_list = Path(episode_dir_list) if episode_dir_list is not None else None
    
    def convert(self) -> LeRobotDataset:
        """Run the conversion and return a ready-to-use LeRobotDataset."""
        # TODO: Lerobot policies only accept low-dimensional features under "observation.state",
        # we should consider customizable low-dim feature keys and flatten the state vector into a single feature.
        if self.episode_dir_list is not None:
            episodes = self._load_episode_dirs_from_file()
        else:
            episodes = self._discover_episode_dirs()
        if len(episodes) == 0:
            raise FileNotFoundError(
                f"No LBM episodes found under '{self.lbm_root}'. "
                "Expected tasks/*/*/sim/bc/teleop/*/diffusion_spartan/episode_*/processed"
            )
        logging.info(f"Found {len(episodes)} episodes.")


        ds_root = None if self.output_root is None else self.output_root / self.repo_id

        if ds_root.exists() and ds_root.is_dir():
            logging.info(f"Dataset already exists at {ds_root}, resuming...")
            dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=ds_root,
                video_backend="torchcodec",
            )
            # Check if the dataset contains all the episodes
            # if not, resuming from the last episode
            if dataset.meta.total_episodes < len(episodes):
                logging.info(f"Dataset contains {dataset.meta.total_episodes} episodes, resuming from the last episode {dataset.meta.total_episodes}")
                episodes = episodes[dataset.meta.total_episodes:]
                features = dataset.features
            else:
                logging.info(f"Dataset contains all the episodes.")
                return dataset
        else:
            logging.info(f"Dataset does not exist at {ds_root}, creating a new dataset")
            sample_obs, _, camera_map, camera_shapes = self._load_episode(episodes[0])
            features, _, _ = self._build_features(sample_obs, camera_map, camera_shapes, self.state_keys)
            dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=features,
                root=ds_root,
                use_videos=self.use_videos,
                image_writer_threads=16,
                video_backend="torchcodec",
            )

        logging.info("Converting %d LBM episodes to LeRobotDataset '%s'", len(episodes), self.repo_id)

        for ep_idx, episode_dir in enumerate(tqdm(episodes, desc="LBM->LeRobot")):
            obs, actions, camera_map, camera_shapes = self._load_episode(episode_dir)
            # Build a map for this episode in case camera metadata differs
            _, obs_feature_map, obs_shape_map = self._build_features(obs, camera_map, camera_shapes, self.state_keys)
            num_frames = min(actions.shape[0], self._get_num_frames(obs))
            skill_name = obs["language_instruction"][0].strip() if "language_instruction" in obs and len(obs["language_instruction"]) > 0 else self._extract_skill_name(episode_dir)

            for frame_idx in range(num_frames):
                frame = {
                    "action": actions[frame_idx].astype(np.float32),
                    "task": skill_name + "\n",
                }

                for obs_key, feature_key in obs_feature_map.items():
                    if feature_key not in features or obs_key not in obs.keys():
                        continue
                    if feature_key.startswith('observation.image'):
                        # Use resize_and_pad to maintain aspect ratio and pad to target shape
                        target_shape = features[feature_key]["shape"]
                        frame[feature_key] = resize_and_pad(obs[obs_key][frame_idx], target_shape)
                        continue
                    frame[feature_key] = obs[obs_key][frame_idx]

                # Flatten and concatenate low-dim states to a single vector observation.state
                if "observation.state" in obs_feature_map:
                    state_data = []
                    for state_key in self.state_keys:
                        data = obs[state_key][frame_idx]
                        if data.ndim > 1:
                            data = data.reshape(-1)
                        data = data.astype(np.float32)
                        state_data.append(data)
                    frame["observation.state"] = np.concatenate(state_data, axis=0)

                dataset.add_frame(frame)

            dataset.save_episode()

        return dataset

    def _load_episode_dirs_from_file(self) -> list[Path]:
        """Load episode directories from a file.
        
        The file should contain one episode path per line. Paths can be:
        - Absolute paths
        - Relative paths (relative to lbm_root)
        - Paths ending with 'processed' or paths that need 'processed' appended
        """
        if not self.episode_dir_list.exists():
            raise FileNotFoundError(f"Episode directory list file not found: {self.episode_dir_list}")
        
        episodes = []
        with self.episode_dir_list.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Try as absolute path first
                episode_path = Path(line)
                if not episode_path.is_absolute():
                    # If relative, assume it's relative to lbm_root
                    episode_path = self.lbm_root / episode_path
                
                # Ensure the path ends with 'processed'
                if episode_path.name != "processed":
                    episode_path = episode_path / "processed"

                # Filter the episode path by the skill types
                if self.skill_types is not None:
                    skill_name = episode_path.parts[-9]
                    if skill_name not in self.skill_types:
                        continue
                
                # Verify the path exists
                if episode_path.exists() and episode_path.is_dir():
                    episodes.append(episode_path)
                else:
                    logging.warning(f"Episode directory not found or invalid: {episode_path}")
        
        return episodes

    def _discover_episode_dirs(self) -> list[Path]:
        if self.skill_types is not None:
            episodes = []
            for skill_type in self.skill_types:
                pattern = f"tasks/{skill_type}/**/diffusion_spartan/episode_*/processed"
                episodes.extend(sorted(self.lbm_root.glob(pattern)))
            return episodes
        else:
            pattern = "**/diffusion_spartan/episode_*/processed"
            return sorted(self.lbm_root.glob(pattern))

    def _load_episode(
        self, processed_dir: Path
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str], dict[str, tuple[int, int, int]]]:
        obs_path = processed_dir / "observations.npz"
        actions_path = processed_dir / "actions.npz"
        if not obs_path.exists() or not actions_path.exists():
            raise FileNotFoundError(f"Missing observations/actions in {processed_dir}")

        obs_npz = np.load(obs_path, allow_pickle=True)
        actions_npz = np.load(actions_path, allow_pickle=True)

        actions = (
            actions_npz["actions"]
            if isinstance(actions_npz, np.lib.npyio.NpzFile) and "actions" in actions_npz.files
            else actions_npz.f.actions
        )

        camera_map = self._camera_name_map(processed_dir.parent)
        camera_shapes = self._camera_shapes(processed_dir, camera_map)
        # Keep streams except depth/label; low-dim states are kept
        obs = {k: obs_npz[k] for k in obs_npz.files if not (k.endswith("_depth") or k.endswith("_label"))}
        return obs, actions, camera_map, camera_shapes

    def _build_features(
        self, observations: dict[str, np.ndarray], camera_map: dict[str, str], camera_shapes: dict[str, tuple[int, int, int]],
        state_keys: Iterable[str] | None = None,
    ) -> tuple[dict, dict[str, str], dict[str, tuple[int, int, int] | None]]:
        if len(observations) == 0:
            raise ValueError("No observation streams found to infer features.")
        
        if state_keys is None:
            state_keys = []
        else:
            state_keys = list(state_keys)
        
        features: dict[str, dict] = {
            "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        }
        obs_feature_map: dict[str, str] = {}
        obs_shape_map: dict[str, tuple[int, int, int] | None] = {}
        obs_state_size: int = 0

        # Target image shape: (480, 640, 3) (H, W, C)
        TARGET_IMAGE_SHAPE = (480, 640, 3)
        
        for cam_id, frames in observations.items():
            # Image streams
            if frames.ndim == 4 or (frames.ndim == 2 and cam_id in camera_shapes):
                semantic = camera_map.get(cam_id, cam_id)
                semantic = _sanitize_name(semantic)
                if self.keep_cameras is not None and semantic not in self.keep_cameras:
                    continue
                # Use target shape for all images
                h, w, c = TARGET_IMAGE_SHAPE
                obs_shape_map[cam_id] = (h, w, c)
                feature_key = f"observation.images.{semantic}"
                features[feature_key] = {
                    "dtype": "video" if self.use_videos else "image",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channels"],
                }
                obs_feature_map[cam_id] = feature_key
                continue

            # Low-dimensional state streams (T, D...) -> flatten to vector
            # If state_keys is provided, only keep the specified keys and concate them into a single vector observation.state
            if frames.ndim >= 2:
                if state_keys and cam_id not in state_keys:
                    continue
                obs_state_size += int(np.prod(frames.shape[1:]))
                continue

        if obs_state_size > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (obs_state_size,),
                "names": None,
            }
            obs_feature_map["observation.state"] = "observation.state"
            obs_shape_map["observation.state"] = (obs_state_size, None)

        # # Language instruction if exists
        # if 'language_instruction' in observations:
        #     features["observation.language"] = {
        #         "dtype": "string",
        #         "shape": (1,),
        #         "names": None,
        #     }
        #     obs_feature_map["language_instruction"] = "observation.language"
        #     obs_shape_map["language_instruction"] = (1, None)

        if not any(k.startswith("observation.images") for k in features):
            raise ValueError(f"No RGB observation streams detected in LBM data. Observation keys: {observations.keys()}.")

        # Ensure default bookkeeping features are present
        return {**features, **DEFAULT_FEATURES}, obs_feature_map, obs_shape_map

    def _camera_name_map(self, episode_dir: Path) -> dict[str, str]:
        metadata_path = episode_dir / "processed" / "metadata.yaml"
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r") as f:
                meta = yaml.safe_load(f)
            mapping = meta.get("camera_id_to_semantic_name", {}) or {}
        except Exception:
            mapping = {}
        return {cam_id: _sanitize_name(name) for cam_id, name in mapping.items()}

    def _camera_shapes(self, processed_dir: Path, camera_map: dict[str, str]) -> dict[str, tuple[int, int, int]]:
        shapes: dict[str, tuple[int, int, int]] = {}
        for cam_id in camera_map:
            info_path = processed_dir / f"images_{cam_id}" / "camera_info.yaml"
            if not info_path.exists():
                continue
            try:
                with info_path.open("r") as f:
                    info = yaml.safe_load(f)
                h = int(info["camera_matrix"]["image_height"])
                w = int(info["camera_matrix"]["image_width"])
                shapes[cam_id] = (h, w, 3)
            except Exception:
                continue
        return shapes

    @staticmethod
    def _get_num_frames(obs: dict[str, np.ndarray]) -> int:
        first_key = next(iter(obs))
        return obs[first_key].shape[0]

    @staticmethod
    def _extract_skill_name(episode_dir: Path) -> str:
        # tasks/{SKILL}/{STATION}/...
        try:
            task_idx = episode_dir.parts.index("tasks") + 1
            task_name = episode_dir.parts[task_idx]
            # cast camel case to lowercase words
            return pascal_to_snake_case(task_name)

        except (ValueError, IndexError):
            return "lbm_task"

