#!/usr/bin/env python3
"""
LBM Eval VLA Policy Wrapper

A policy wrapper for LBM Eval that supports multiple VLA models (pi0.5, groot, smolvla).
This wrapper converts between LBM Eval's MultiarmObservation/PosesAndGrippers format
and LeRobot's observation/action format, and serves inference via gRPC.

IMPORTANT: Format Consistency with Training Dataset
- Observation state format: [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1)]
  (left first, right second - matches training dataset)
- Action format: [right_xyz(3), right_rot6d(6), left_xyz(3), left_rot6d(6), right_gripper(1), left_gripper(1)]
  (right first, left second - matches training dataset)
Note: The order differs between state and action to match the training dataset format.

Usage:
    python -m lerobot.scripts.lbm_replay_policy_server \
        --repo-id <repo_id> \
        --dataset-root <dataset_root> \
        --episodes <episode_number> \
        --server-uri localhost:50051
"""

import argparse
import copy
import logging
import math
import threading
import time
from typing import Optional
import uuid
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

# LBM Eval imports (external dependency)
try:
    from grpc_workspace.lbm_policy_server import (
        LbmPolicyServerConfig,
        run_policy_server,
    )
    from grpc_workspace.lbm_policy_conversions import array_to_rotation_matrix
    from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
    from robot_gym.policy import Policy, PolicyMetadata
except ImportError as e:
    raise ImportError(
        "Missing required dependencies for LBM Eval. "
        "Please install robot_gym and grpc_workspace packages. "
        f"Original error: {e}"
    )

# LeRobot imports
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEBUG = True
DEBUG_DIR = Path("outputs/lbm_replay_policy_server").resolve()
DEBUG_DIR.mkdir(parents=False, exist_ok=True)


logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rot6d: 6D rotation vector of shape (6,) or (N, 6)
        
    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3)
    """
    if len(rot6d.shape) == 1:
        rot6d = rot6d[None, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]
    
    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
    
    # Orthogonalize and normalize second column
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)
    
    # Third column via cross product
    b3 = np.cross(b1, b2, axis=-1)
    
    # Stack to form rotation matrix
    rotation_matrix = np.stack([b1, b2, b3], axis=-2)  # (N, 3, 3)
    
    if squeeze_output:
        rotation_matrix = rotation_matrix[0]
    
    return rotation_matrix


def rotation_matrix_to_rot6d(rot_matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rot_matrix: Rotation matrix of shape (3, 3) or (N, 3, 3)
        
    Returns:
        6D rotation vector of shape (6,) or (N, 6)
    """
    if len(rot_matrix.shape) == 2:
        rot_matrix = rot_matrix[None, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_shape = rot_matrix.shape[:-2]
    rot6d = rot_matrix[..., :2, :].copy().reshape(batch_shape + (6,))
    
    if squeeze_output:
        rot6d = rot6d[0]
    
    return rot6d



IMAGE_SIZE = (480, 640, 3)

def resize_and_pad_image(image: np.ndarray) -> np.ndarray:
    """
    Resize and pad an image to the target size (IMAGE_SIZE) with zero padding,
    keeping the aspect ratio unchanged (no stretched distortion).
    """
    target_h, target_w, target_c = IMAGE_SIZE
    h, w = image.shape[:2]

    # 计算缩放比例，保持长宽比
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    if image.shape[-1] != target_c:
        raise ValueError(f"Input image has {image.shape[-1]} channels, expected {target_c}")
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建全零的目标尺寸图像
    out = np.zeros((target_h, target_w, target_c), dtype=image.dtype)

    # 计算在目标图像中心位置安放resized图像的起始点
    y_start = (target_h - new_h) // 2
    x_start = (target_w - new_w) // 2

    # 拷贝缩放后的图片到输出图片
    out[y_start:y_start+new_h, x_start:x_start+new_w, :] = resized
    return out

def convert_multiarm_observation_to_lerobot(
    observation: MultiarmObservation,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Convert MultiarmObservation from LBM Eval to LeRobot observation format.
    
    Args:
        observation: MultiarmObservation from robot_gym
        device: PyTorch device to place tensors on
        
    Returns:
        Dictionary with LeRobot-formatted observation tensors
    """
    lerobot_obs = {}

    # Extract images from observation.visuo (camera images are stored here)
    if hasattr(observation, "visuo") and observation.visuo:
        available_cameras = list(observation.visuo.keys())
        logger.debug(f"Available cameras in observation.visuo: {available_cameras}")
        
        for camera_name, camera_image_set in observation.visuo.items():
            # Extract RGB image from CameraImageSet
            rgb_image = None
            if hasattr(camera_image_set, "rgb"):
                rgb_image_obj = camera_image_set.rgb
                if hasattr(rgb_image_obj, "array"):
                    rgb_image = rgb_image_obj.array
                elif isinstance(rgb_image_obj, np.ndarray):
                    rgb_image = rgb_image_obj
            elif isinstance(camera_image_set, np.ndarray):
                rgb_image = camera_image_set
            
            if rgb_image is not None:
                # Resize and pad image to target size
                rgb_image = resize_and_pad_image(rgb_image)
                # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
                if isinstance(rgb_image, np.ndarray):
                    image_tensor = torch.from_numpy(rgb_image.copy()).float()
                else:
                    image_tensor = torch.tensor(rgb_image).float()
                
                # Normalize to [0, 1] if in [0, 255] range
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor / 255.0
                
                # Convert (H, W, C) to (C, H, W)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
                
                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                image_tensor = image_tensor.unsqueeze(0)
                
                # Move to device
                image_tensor = image_tensor.to(device)
                
                # Use the camera name as-is (should match policy expectations)
                image_key = f"{OBS_IMAGES}.{camera_name}"
                lerobot_obs[image_key] = image_tensor
                logger.debug(f"Added image with key: {image_key}, shape: {image_tensor.shape}")
            else:
                logger.warning(f"Could not extract RGB image from camera {camera_name}")
    else:
        logger.warning("No images found in observation.visuo")
    
    # Extract robot state from observation.robot.actual
    if hasattr(observation, "robot") and hasattr(observation.robot, "actual"):
        actual = observation.robot.actual
        
        # Extract robot state if available
        # Try to get poses and construct state vector
        # NOTE: Dataset format has left first, then right in observation state
        # Expected format: [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1)] = 20 dims
        # This matches the training dataset format (left first, right second)
        if hasattr(actual, "poses") and actual.poses:
            state_components = []
            available_robot_names = list(actual.poses.keys())
            
            # Determine robot order: left first, then right (to match dataset format)
            # This is different from action format which is right first, then left
            robot_order = []
            left_robot = None
            right_robot = None
            
            # Try to identify which robot is "left" and which is "right"
            for name in available_robot_names:
                name_lower = name.lower()
                if "left" in name_lower:
                    left_robot = name
                elif "right" in name_lower:
                    right_robot = name
            
            # Build order: left first, then right (to match dataset observation format)
            if left_robot and right_robot:
                robot_order = [left_robot, right_robot]
                # robot_order = [right_robot, left_robot]  # debug
                logger.debug(f"State: Using robot order {robot_order} (left first, right second) to match dataset format")
            elif len(available_robot_names) == 2:
                # Fallback: use sorted order (left comes before right alphabetically)
                robot_order = sorted(available_robot_names)
                logger.warning(
                    f"Could not identify left/right robots for state. Using sorted order: {robot_order}. "
                    "Assuming first is left, second is right to match dataset format."
                )
            else:
                # Single robot or unexpected number
                robot_order = sorted(available_robot_names)
                logger.warning(
                    f"Unexpected number of robots: {len(available_robot_names)}. "
                    f"Using sorted order: {robot_order}"
                )
            
            robot_names = robot_order
            
            # Process each robot in the determined order (left first, then right)
            for robot_name in robot_names:
                pose = actual.poses[robot_name]
                # Extract translation (xyz) - 3 dims
                if hasattr(pose, "translation"):
                    xyz = pose.translation()
                    if isinstance(xyz, np.ndarray):
                        # Make writable copy to avoid PyTorch warning
                        xyz = xyz.copy()
                        state_components.append(torch.from_numpy(xyz).float())
                    else:
                        state_components.append(torch.tensor(xyz, dtype=torch.float32))
                
                # Extract rotation and convert to rot6d (6 dims) to match action format
                if hasattr(pose, "rotation"):
                    rotation = pose.rotation()
                    # Try to get rotation matrix and convert to rot6d
                    rot_matrix = None
                    if hasattr(rotation, "as_matrix"):
                        rot_matrix = rotation.as_matrix()
                    elif hasattr(rotation, "rotation_matrix"):
                        rot_matrix = rotation.rotation_matrix()
                    elif hasattr(rotation, "matrix"):
                        rot_matrix = rotation.matrix()
                    
                    if rot_matrix is not None:
                        if isinstance(rot_matrix, np.ndarray):
                            rot_matrix = rot_matrix.copy()  # Make writable
                        else:
                            rot_matrix = np.array(rot_matrix)
                        # Convert rotation matrix to rot6d
                        rot6d = rotation_matrix_to_rot6d(rot_matrix)
                        state_components.append(torch.from_numpy(rot6d).float())
                    else:
                        # Fallback: try to get rotation matrix from scipy Rotation if available
                        try:
                            from scipy.spatial.transform import Rotation as ScipyRotation
                            if hasattr(rotation, "as_rotvec"):
                                rotvec = rotation.as_rotvec()
                                scipy_rot = ScipyRotation.from_rotvec(rotvec)
                                rot_matrix = scipy_rot.as_matrix()
                                rot6d = rotation_matrix_to_rot6d(rot_matrix)
                                state_components.append(torch.from_numpy(rot6d).float())
                            else:
                                logger.warning(
                                    f"Could not extract rotation matrix for {robot_name}. "
                                    "Using zero rotation (6D)."
                                )
                                state_components.append(torch.zeros(6, dtype=torch.float32))
                        except ImportError:
                            logger.warning(
                                f"Could not extract rotation matrix for {robot_name}. "
                                "scipy not available. Using zero rotation (6D)."
                            )
                            state_components.append(torch.zeros(6, dtype=torch.float32))
            
            # Extract gripper states in the same order as robots
            # Always add gripper values (default to 0.0 if missing) to ensure 20 dims
            for robot_name in robot_names:
                gripper_val = None
                if hasattr(actual, "grippers") and actual.grippers:
                    # Try exact match first
                    if robot_name in actual.grippers:
                        gripper_val = actual.grippers[robot_name]
                    else:
                        # Try with "_hand" suffix (e.g., "right::panda" -> "right::panda_hand")
                        gripper_key = f"{robot_name}_hand"
                        if gripper_key in actual.grippers:
                            gripper_val = actual.grippers[gripper_key]
                        else:
                            # Try to find any gripper key that starts with the robot name
                            for key in actual.grippers.keys():
                                if key.startswith(robot_name):
                                    gripper_val = actual.grippers[key]
                                    break
                
                if gripper_val is not None:
                    if isinstance(gripper_val, (int, float)):
                        state_components.append(torch.tensor([float(gripper_val)], dtype=torch.float32))
                    elif isinstance(gripper_val, np.ndarray):
                        # Make writable copy
                        gripper_val = gripper_val.copy()
                        if gripper_val.ndim == 0:
                            gripper_val = np.array([float(gripper_val)])
                        state_components.append(torch.from_numpy(gripper_val).float())
                    else:
                        state_components.append(torch.tensor([float(gripper_val)], dtype=torch.float32))
                else:
                    # Default gripper value if missing
                    logger.debug(f"Gripper value missing for {robot_name}, using default 0.0")
                    state_components.append(torch.tensor([0.0], dtype=torch.float32))
            
            if state_components:
                state_tensor = torch.cat(state_components, dim=0)
                # Ensure we have exactly 20 dimensions
                if state_tensor.shape[0] != 20:
                    logger.warning(
                        f"Observation state dimension mismatch: expected 20, got {state_tensor.shape[0]}. "
                        f"Robot names (order): {robot_names}, Components: {[c.shape for c in state_components]}. "
                        f"Expected format (dataset): [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1)]"
                    )
                    # Pad or truncate to 20 dims if needed
                    if state_tensor.shape[0] < 20:
                        padding = torch.zeros(20 - state_tensor.shape[0], dtype=torch.float32)
                        state_tensor = torch.cat([state_tensor, padding], dim=0)
                        logger.warning(f"Padded state tensor to 20 dims")
                    else:
                        state_tensor = state_tensor[:20]
                        logger.warning(f"Truncated state tensor to 20 dims")
                else:
                    # Log the actual state structure for debugging
                    # Format: [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1)]
                    logger.debug(
                        f"State tensor constructed successfully. "
                        f"Robot order: {robot_names} (left first, right second to match dataset), "
                        f"Format: [{robot_names[0]}_xyz(3), {robot_names[0]}_rot6d(6), "
                        f"{robot_names[1]}_xyz(3), {robot_names[1]}_rot6d(6), "
                        f"{robot_names[0]}_gripper(1), {robot_names[1]}_gripper(1)]"
                    )
                
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                state_tensor = state_tensor.to(device)
                lerobot_obs[OBS_STATE] = state_tensor
    
    # Extract language instruction if available
    if hasattr(observation, "language_instruction") and observation.language_instruction:
        lerobot_obs["task"] = observation.language_instruction
    
    return lerobot_obs


def convert_action_to_poses_and_grippers(
    action_tensor: torch.Tensor,
    robot_names: list[str] | None = None,
    reference_poses: dict | None = None,
    reference_grippers: dict | None = None,
) -> PosesAndGrippers:
    """
    Convert LeRobot action tensor (20-dim) to PosesAndGrippers format.
    
    Expected action format: [right_xyz(3), right_rot6d(6), left_xyz(3), left_rot6d(6), right_gripper(1), left_gripper(1)]
    
    Args:
        action_tensor: Action tensor of shape (batch_size, 20) or (20,)
        robot_names: List of robot names, default ["right", "left"]
        reference_poses: Optional dictionary of reference Pose objects to copy from
        
    Returns:
        PosesAndGrippers object with poses and grippers dictionaries
    """
    from pydrake.math import RigidTransform
    
    if robot_names is None:
        robot_names = ["right", "left"]
    
    # Remove batch dimension if present
    if action_tensor.ndim > 1:
        action_tensor = action_tensor.squeeze(0)
    
    # Convert to numpy
    action_np = action_tensor.detach().cpu().numpy()
    
    if len(action_np) != 20:
        raise ValueError(
            f"Expected action dimension 20, got {len(action_np)}. "
            "Action format: [right_xyz(3), right_rot6d(6), left_xyz(3), left_rot6d(6), right_gripper(1), left_gripper(1)]"
        )
    
    # Parse action components
    right_xyz = action_np[0:3]
    right_rot6d = action_np[3:9]
    left_xyz = action_np[9:12]
    left_rot6d = action_np[12:18]
    right_gripper = float(action_np[18])
    left_gripper = float(action_np[19])
    # left_xyz = action_np[0:3]
    # left_rot6d = action_np[3:9]
    # right_xyz = action_np[9:12]
    # right_rot6d = action_np[12:18]
    # left_gripper = float(action_np[18]) # debug
    # right_gripper = float(action_np[19]) # debug
    
    # Convert 6D rotation to rotation matrix
    right_rot_matrix = rot6d_to_rotation_matrix(right_rot6d)
    left_rot_matrix = rot6d_to_rotation_matrix(left_rot6d)
    
    # Create Pose objects
    poses = {}
    grippers = {}
    
    # Helper function to create or update a pose
    def create_or_update_pose(
        robot_name: str,
        translation: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> RigidTransform:
        """Create a new Pose or copy and update an existing one."""
        if reference_poses and robot_name in reference_poses:
            # convert rotation matrix to pydrake rotation
            rotation = array_to_rotation_matrix(rotation_matrix.flatten())
            pose = RigidTransform(R=rotation, p=translation)
            
            # # Update rotation - try different methods based on Pose API
            # if hasattr(pose, "set_rotation_matrix"):
            #     pose.set_rotation_matrix(rotation_matrix)
        # elif hasattr(pose, "set_rotation"):
        #     # Try using scipy Rotation object
        #     try:
        #         from scipy.spatial.transform import Rotation  # type: ignore
        #         rotation = Rotation.from_matrix(rotation_matrix)
        #         pose.set_rotation(rotation)
        #     except (ImportError, AttributeError):
        #         # If scipy not available or set_rotation doesn't accept Rotation,
        #         # try to get rotation from pose and update it
        #         if hasattr(pose, "rotation"):
        #             current_rotation = pose.rotation()
        #             # Try to update rotation in place if possible
        #             logger.warning(
        #                 f"Could not set rotation for {robot_name} pose. "
        #                 "Rotation matrix conversion may not be fully supported."
        #             )
        #     else:
        #         logger.warning(
        #             f"Pose object for {robot_name} does not have set_rotation_matrix or set_rotation methods. "
        #             "Rotation may not be updated correctly."
        #         )
        # else:
        #     # Create new pose
        #     pose = RigidTransform()
        #     pose.set_translation(translation)
        #     # Set rotation
        #     if hasattr(pose, "set_rotation_matrix"):
        #         pose.set_rotation_matrix(rotation_matrix)
        #     elif hasattr(pose, "set_rotation"):
        #         try:
        #             from scipy.spatial.transform import Rotation  # type: ignore
        #             rotation = Rotation.from_matrix(rotation_matrix)
        #             pose.set_rotation(rotation)
        #         except (ImportError, AttributeError):
        #             logger.warning(
        #                 f"Could not set rotation for new {robot_name} pose. "
        #                 "Rotation may not be initialized correctly."
        #             )
        else:
            raise ValueError(f"Could not create or update pose for {robot_name}, reference_poses: {reference_poses}.")
        
        return pose
    
    # Create poses for both arms
    poses[robot_names[0]] = create_or_update_pose(
        robot_names[0], right_xyz, right_rot_matrix
    )
    poses[robot_names[1]] = create_or_update_pose(
        robot_names[1], left_xyz, left_rot_matrix
    )
    
    # Map robot names to gripper names
    # Try to infer gripper names from reference_grippers if provided
    right_gripper_name = None
    left_gripper_name = None
    
    if reference_grippers:
        # Find gripper keys that correspond to each robot
        # Match by checking if gripper key starts with robot name
        for gripper_key in reference_grippers.keys():
            if gripper_key.startswith(robot_names[0]) and right_gripper_name is None:
                right_gripper_name = gripper_key
            elif gripper_key.startswith(robot_names[1]) and left_gripper_name is None:
                left_gripper_name = gripper_key
    
    # Fallback: append '_hand' to robot name if not found in reference_grippers
    if right_gripper_name is None:
        right_gripper_name = f"{robot_names[0]}_hand"
    if left_gripper_name is None:
        left_gripper_name = f"{robot_names[1]}_hand"
    
    grippers[right_gripper_name] = right_gripper
    grippers[left_gripper_name] = left_gripper
    
    return PosesAndGrippers(poses=poses, grippers=grippers)


def _get_policy_metadata(model_type: str, checkpoint_path: str) -> PolicyMetadata:
    """Get policy metadata for the wrapper."""
    return PolicyMetadata(
        name=f"LeRobotVLA-{model_type}",
        skill_type="Undefined",
        checkpoint_path=checkpoint_path,
        git_repo="lerobot",
        git_sha="Undefined",
    )


class ReplayPolicy(Policy):
    """A policy wrapper that uses LeRobot VLA models for LBM Eval.
    
    Supports both RTC-enabled and non-RTC modes.
    """

    def __init__(
        self,
        repo_id: str,
        dataset_root: str,
        episode: int,
    ):
        """Initialize ReplayPolicy wrapper.
        
        Args:
            repo_id: Repository ID
            dataset_root: Root directory of the dataset
            episode: Episode to replay
        """
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.episode = episode
        
        # Load dataset
        self.dataset = LeRobotDataset(repo_id, dataset_root, episodes=[episode],
            video_backend="pyav")

        # Get feature keys
        self.feature_keys = self.dataset.features
        self.expected_image_keys = set([k for k in self.feature_keys if k.startswith(OBS_IMAGES + ".")])
        
        self.device = torch.device("cpu")
        
        self.reset()

    def reset(self):
        """Reset policy state when environment resets."""
        self.current_index = 0
        
    def get_policy_metadata(self):
        """Get policy metadata."""
        return _get_policy_metadata(self.repo_id, self.dataset_root, self.episode)

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        """Perform one step of inference.
        
        Args:
            observation: MultiarmObservation from LBM Eval
            
        Returns:
            PosesAndGrippers with predicted poses and grippers
        """
        # Convert observation to LeRobot format
        lerobot_obs = convert_multiarm_observation_to_lerobot(observation, self.device)
        dataset_sample = self.dataset[self.current_index]
        obs_image_keys = [k for k in lerobot_obs.keys() if k.startswith(OBS_IMAGES + ".")]
        logger.debug(f"Task: {observation.language_instruction}")
        logger.debug(f"Observation keys before preprocessing: {list(lerobot_obs.keys())}")
        logger.debug(f"Dataset keys before preprocessing: {list(dataset_sample.keys())}")
        logger.debug(f"Policy expected image keys: {sorted(self.expected_image_keys)}")

        # Check if image keys match
        if obs_image_keys and self.expected_image_keys:
            missing_keys = self.expected_image_keys - set(obs_image_keys)
            extra_keys = set(obs_image_keys) - self.expected_image_keys
            if missing_keys or extra_keys:
                logger.warning(
                    f"Image key mismatch - Missing: {sorted(missing_keys)}, Extra: {sorted(extra_keys)}"
                )
        
        # Check if current index is within dataset length
        if self.current_index >= len(self.dataset):
            logger.warning(f"Current index {self.current_index} is greater than dataset length {len(self.dataset)}")
            self.current_index = len(self.dataset) - 1

        # Save observation and dataset images
        if DEBUG:
            for key in obs_image_keys:
                image = lerobot_obs[key].squeeze(0).cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = image * 255
                image = image.astype(np.uint8)
                image = Image.fromarray(image)
                image.save(DEBUG_DIR / f"observation_{key}_ep{self.episode}_idx{self.current_index}.png")
            for key in self.expected_image_keys:
                image = dataset_sample[key].squeeze(0).cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = image * 255
                image = image.astype(np.uint8)
                image = Image.fromarray(image)
                image.save(DEBUG_DIR / f"dataset_{key}_ep{self.episode}_idx{self.current_index}.png")
        
        # Compare current observation state with expected state
        obs_state = lerobot_obs[OBS_STATE]
        expected_state = dataset_sample[OBS_STATE]
        logger.debug(f"Difference between expected and actual state: {obs_state - expected_state}")
        
        action_tensor = dataset_sample[ACTION]
        
        
        # Convert action to PosesAndGrippers format
        # Extract robot names, reference poses, and gripper keys from observation if available
        # IMPORTANT: robot_names order for action must be right first, then left
        # This matches action format: [right_xyz(3), right_rot6d(6), left_xyz(3), left_rot6d(6), right_gripper(1), left_gripper(1)]
        # Note: This is different from state format which is left first, then right
        robot_names = None
        reference_poses = None
        reference_grippers = None
        if hasattr(observation, "robot") and hasattr(observation.robot, "actual"):
            actual = observation.robot.actual
            if hasattr(actual, "poses"):
                # For action conversion, we need right first, then left (to match action format)
                available_robot_names = list(actual.poses.keys())
                right_robot = None
                left_robot = None
                
                for name in available_robot_names:
                    name_lower = name.lower()
                    if "right" in name_lower:
                        right_robot = name
                    elif "left" in name_lower:
                        left_robot = name
                
                if right_robot and left_robot:
                    robot_names = [right_robot, left_robot]  # right first, left second (for action format)
                    # robot_names = [left_robot, right_robot] # debug
                    logger.debug(f"Action: Using robot order {robot_names} (right first, left second) to match action format")
                else:
                    # Fallback: try to determine order
                    # If sorted order gives left first, reverse it to get right first
                    sorted_names = sorted(available_robot_names)
                    if len(sorted_names) == 2 and "left" in sorted_names[0].lower():
                        robot_names = [sorted_names[1], sorted_names[0]]  # Reverse to get right first
                    else:
                        robot_names = sorted_names
                    logger.warning(
                        f"Could not identify right/left robots for action conversion. "
                        f"Using order: {robot_names}. Assuming first is right, second is left."
                    )
                
                reference_poses = actual.poses
            if hasattr(actual, "grippers"):
                reference_grippers = actual.grippers
        
        poses_and_grippers = convert_action_to_poses_and_grippers(
            action_tensor, 
            robot_names=robot_names, 
            reference_poses=reference_poses,
            reference_grippers=reference_grippers
        )
        self.current_index += 1 # increment current index

        logging.debug("One step completed")
        
        return poses_and_grippers

    


class ReplayPolicyBatch(Policy):
    """Batch policy wrapper for gRPC interface.
    
    Supports multiple concurrent evaluations by sharing a single policy instance
    (to save GPU memory) while maintaining separate state per UUID.
    
    Thread Safety:
        - The shared policy instance is safe for concurrent inference because:
          1. PyTorch models in inference mode are thread-safe (read-only operations)
          2. We use predict_action_chunk() which doesn't rely on policy's internal state
          3. All session-specific state (action queues, RTC state) is managed in VLAPolicy wrappers
        - Each UUID has its own VLAPolicy wrapper with independent state, preventing conflicts.
    """

    def __init__(
        self,
        repo_id: str,
        dataset_root: str,
        episodes: list[int],
    ):
        """Initialize batch policy wrapper.
        
        Args:
            policy_class: LeRobot policy class to instantiate
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            model_type: Model type name (pi05, groot, smolvla)
            rtc_enabled_override: If True/False, override RTC setting from config.
                                 If None, use config setting.
            compile_model_override: If True/False, override compile_model setting from config.
                                   If None, use config setting. Only applies to pi05 and pi0 models.
        """
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.episodes = episodes
        
        # Internal UUID for non-batch interface
        self._internal_uuid = uuid.uuid4()
        
        # Mapping from UUID to VLAPolicy wrappers (each maintains its own state)
        # Note: All wrappers share the same underlying policy instance
        self._sub_policies: dict[uuid.UUID, ReplayPolicy] = {}
        
        logger.info(f"ReplayPolicyBatch initialized for episodes: {episodes}")

    def reset(self, seed: int | None = None, options=None):
        """Reset internal policy state."""
        self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(
        self, seeds: dict[uuid.UUID, int | None], options=None
    ) -> None:
        """Reset batch of policies.
        
        Args:
            seeds: Dictionary mapping UUIDs to seeds (ignored for VLA models)
            options: Optional reset options
        """
        for one_uuid, one_seed in seeds.items():
            if one_seed is not None:
                warnings.warn(f"We ignore the seed for {one_uuid}!")
            
            # Create or reset policy wrapper for this UUID
            if one_uuid not in self._sub_policies:
                # Create new wrapper that shares the same policy instance
                # Each wrapper maintains its own state (action queue, RTC state, etc.)
                replay_policy = ReplayPolicy(
                    repo_id=self.repo_id,
                    dataset_root=self.dataset_root,
                    episode=self.episodes[len(self._sub_policies)],
                )
                self._sub_policies[one_uuid] = replay_policy
            else:
                # Reset existing wrapper's state
                self._sub_policies[one_uuid].reset()

    def get_policy_metadata(self):
        """Get policy metadata."""
        return _get_policy_metadata(self.repo_id, self.dataset_root)

    def step(self, observation):
        """Single step (delegates to batch interface)."""
        batch_actions = self.step_batch({self._internal_uuid: observation})
        return batch_actions[self._internal_uuid]

    def step_batch(
        self, observations: dict[uuid.UUID, MultiarmObservation]
    ) -> dict[uuid.UUID, PosesAndGrippers]:
        """Perform batch inference.
        
        Args:
            observations: Dictionary mapping UUIDs to MultiarmObservation
            
        Returns:
            Dictionary mapping UUIDs to PosesAndGrippers
        """
        batch_actions = {}
        for one_uuid, observation in observations.items():
            if one_uuid not in self._sub_policies:
                # Create policy if it doesn't exist
                self.reset_batch({one_uuid: None})
            
            sub_policy = self._sub_policies[one_uuid]
            batch_actions[one_uuid] = sub_policy.step(observation)

        logging.debug("One batch step completed")
        
        return batch_actions


def main():
    """Main entry point for the policy server."""
    parser = argparse.ArgumentParser(
        description="LBM Eval VLA Policy Wrapper Server"
    )
    
    # Add LBM policy server config arguments
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    
    # Add custom arguments
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="Episodes to replay",
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not Path(args.dataset_root).exists():
        logger.error(f"Dataset root directory {args.dataset_root} does not exist")
        raise FileNotFoundError(f"Dataset root directory {args.dataset_root} does not exist")
    
    logger.info(f"Loading dataset from {args.dataset_root} for episodes: {args.episodes}")
    
    # Create batch policy wrapper
    policy = ReplayPolicyBatch(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
    )
    
    logger.info("Policy wrapper initialized successfully")
    logger.info(f"Starting server on {args.server_uri}")
    
    # Start gRPC server
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()

