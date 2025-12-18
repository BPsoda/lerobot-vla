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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

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
    ) -> None:
        self.lbm_root = Path(lbm_root)
        self.output_root = Path(output_root) if output_root is not None else None
        self.repo_id = repo_id
        self.fps = fps
        self.use_videos = use_videos
        self.keep_cameras = list(keep_cameras) if keep_cameras is not None else None
        self.state_keys = list(state_keys) if state_keys is not None else None
        self.skill_types = list(skill_types) if skill_types is not None else None
    
    def convert(self) -> LeRobotDataset:
        """Run the conversion and return a ready-to-use LeRobotDataset."""
        # TODO: Lerobot policies only accept low-dimensional features under "observation.state",
        # we should consider customizable low-dim feature keys and flatten the state vector into a single feature.
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
            )

        logging.info("Converting %d LBM episodes to LeRobotDataset '%s'", len(episodes), self.repo_id)

        for ep_idx, episode_dir in enumerate(tqdm(episodes, desc="LBM->LeRobot")):
            obs, actions, camera_map, camera_shapes = self._load_episode(episode_dir)
            # Build a map for this episode in case camera metadata differs
            _, obs_feature_map, obs_shape_map = self._build_features(obs, camera_map, camera_shapes, self.state_keys)
            num_frames = min(actions.shape[0], self._get_num_frames(obs))
            skill_name = obs["language_instruction"][frame_idx].strip() if "language_instuction" in obs else self._extract_skill_name(episode_dir)

            for frame_idx in range(num_frames):
                frame = {
                    "action": actions[frame_idx].astype(np.float32),
                    "task": skill_name + "\n",
                }

                for obs_key, feature_key in obs_feature_map.items():
                    if feature_key not in features or obs_key not in obs.keys():
                        continue
                    if feature_key.startswith('observation.image'):
                        # check if the image size matches the feature shape
                        # if not, resize the image to the feature shape
                        if obs[obs_key][frame_idx].shape != features[feature_key]["shape"]:
                            frame[feature_key] = resize_image(obs[obs_key][frame_idx], features[feature_key]["shape"])
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

        dataset._close_writer()
        dataset.meta._close_writer()
        return dataset

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

        for cam_id, frames in observations.items():
            # Image streams
            if frames.ndim == 4 or (frames.ndim == 2 and cam_id in camera_shapes):
                semantic = camera_map.get(cam_id, cam_id)
                semantic = _sanitize_name(semantic)
                if self.keep_cameras is not None and semantic not in self.keep_cameras:
                    continue
                if frames.ndim == 4:
                    h, w, c = frames.shape[1:]
                else:
                    h, w, c = camera_shapes[cam_id]
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

