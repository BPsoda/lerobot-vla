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
    python -m lerobot.scripts.lbm_eval_policy_server \
        --model-type pi05 \
        --checkpoint-path /path/to/checkpoint \
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
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import get_safe_torch_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    rotation_matrix = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)
    
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
    
    # Extract first two columns
    col1 = rot_matrix[:, :, 0]  # (N, 3)
    col2 = rot_matrix[:, :, 1]  # (N, 3)
    
    # Concatenate to form 6D representation
    rot6d = np.concatenate([col1, col2], axis=-1)  # (N, 6)
    
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


class VLAPolicy(Policy):
    """A policy wrapper that uses LeRobot VLA models for LBM Eval.
    
    Supports both RTC-enabled and non-RTC modes.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor,
        postprocessor,
        device: torch.device,
        rtc_enabled_override: bool | None = None,
        policy_lock: Optional[threading.Lock] = None,
        task_instruction_override: str | None = None,
        n_action_step: int = 1,
    ):
        """Initialize VLA policy wrapper.
        
        Args:
            policy: LeRobot PreTrainedPolicy instance
            preprocessor: Preprocessor pipeline
            postprocessor: Postprocessor pipeline
            device: Device for inference
            rtc_enabled_override: If True/False, override RTC setting from config.
                                 If None, use config setting.
            policy_lock: Optional lock to protect shared policy access (for thread safety).
                        If None, no locking is performed (assumes single-threaded or external locking).
            task_instruction_override: If provided, overrides the task instruction from observation.
                                      If None, uses observation.language_instruction.
            n_action_step: Number of action steps to add to queue from the front of action chunk
                           after each policy predict. Default is 1.
        """
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.policy_lock = policy_lock
        self.task_instruction_override = task_instruction_override
        self.n_action_step = n_action_step
        self.model_type = getattr(policy, "name", "unknown")
        self.checkpoint_path = getattr(policy.config, "pretrained_path", "unknown")
        
        # Get expected image feature keys from policy config for debugging
        self.expected_image_keys = set()
        if hasattr(policy, "config") and hasattr(policy.config, "input_features"):
            for key, feature in policy.config.input_features.items():
                if key.startswith(OBS_IMAGES + "."):
                    self.expected_image_keys.add(key)
        logger.debug(f"Policy expects image keys: {sorted(self.expected_image_keys)}")
        
        # Check if RTC is enabled (use override if provided, otherwise check config)
        if rtc_enabled_override is not None:
            self.rtc_enabled = rtc_enabled_override
            # If overriding, also update the policy config if possible
            if hasattr(self.policy, "config") and hasattr(self.policy.config, "rtc_config"):
                if self.policy.config.rtc_config is not None:
                    self.policy.config.rtc_config.enabled = rtc_enabled_override
                elif rtc_enabled_override:
                    # If RTC config doesn't exist but we want to enable it, warn
                    logger.warning(
                        "RTC override is True but policy config has no rtc_config. "
                        "RTC may not work correctly."
                    )
        else:
            self.rtc_enabled = self._check_rtc_enabled()
        
        # RTC-related state
        self._action_queue = deque(maxlen=100)  # Queue for actions when not using RTC
        self._original_action_queue = None  # Original actions for RTC
        self._last_inference_time = None
        self._inference_delays = deque(maxlen=10)  # Track recent inference delays
        
        self.reset()

    def _check_rtc_enabled(self) -> bool:
        """Check if RTC is enabled in the policy config."""
        if hasattr(self.policy, "_rtc_enabled"):
            return self.policy._rtc_enabled()
        if hasattr(self.policy, "config") and hasattr(self.policy.config, "rtc_config"):
            return (
                self.policy.config.rtc_config is not None
                and self.policy.config.rtc_config.enabled
            )
        return False

    def reset(self):
        """Reset policy state when environment resets."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        
        # Reset RTC-related state
        self._action_queue.clear()
        self._original_action_queue = None
        self._last_inference_time = None
        self._inference_delays.clear()

    def get_policy_metadata(self):
        """Get policy metadata."""
        return _get_policy_metadata(self.model_type, self.checkpoint_path)

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        """Perform one step of inference.
        
        Args:
            observation: MultiarmObservation from LBM Eval
            
        Returns:
            PosesAndGrippers with predicted poses and grippers
        """
        # Convert observation to LeRobot format
        lerobot_obs = convert_multiarm_observation_to_lerobot(observation, self.device)
        
        # Override task instruction if specified
        if self.task_instruction_override is not None:
            lerobot_obs["task"] = self.task_instruction_override
            logger.debug(f"Task instruction overridden: {self.task_instruction_override} (original: {observation.language_instruction})")
        else:
            logger.debug(f"Task: {observation.language_instruction}")
        
        obs_image_keys = [k for k in lerobot_obs.keys() if k.startswith(OBS_IMAGES + ".")]
        logger.debug(f"Observation keys before preprocessing: {list(lerobot_obs.keys())}")
        logger.debug(f"Observation image keys: {obs_image_keys}")
        logger.debug(f"Policy expected image keys: {sorted(self.expected_image_keys)}")
        
        # Check if image keys match
        if obs_image_keys and self.expected_image_keys:
            missing_keys = self.expected_image_keys - set(obs_image_keys)
            extra_keys = set(obs_image_keys) - self.expected_image_keys
            if missing_keys or extra_keys:
                logger.debug(
                    f"Image key mismatch - Missing: {sorted(missing_keys)}, Extra: {sorted(extra_keys)}"
                )
            # remove image keys that are not in the expected image keys
            for key in obs_image_keys:
                if key not in self.expected_image_keys:
                    del lerobot_obs[key]
        
        # Apply preprocessor
        lerobot_obs = self.preprocessor(lerobot_obs)
        obs_image_keys_after = [k for k in lerobot_obs.keys() if k.startswith(OBS_IMAGES + ".")]
        logger.debug(f"Observation keys after preprocessing: {list(lerobot_obs.keys())}")
        logger.debug(f"Observation image keys after preprocessing: {obs_image_keys_after}")
        
        # Run policy inference
        # NOTE: We use predict_action_chunk even for non-RTC mode to avoid
        # conflicts with shared policy's internal state (e.g., _action_queue).
        # We manage action queue in VLAPolicy wrapper instead.
        # Use lock to protect shared policy access if lock is provided
        # This prevents deadlocks when multiple threads access the shared policy concurrently
        from contextlib import nullcontext
        
        lock_context = self.policy_lock if self.policy_lock else nullcontext()
        with lock_context:
            logger.debug("Lock acquired, starting inference")
            with torch.no_grad():
                if self.rtc_enabled:
                    # Use RTC mode: predict_action_chunk with RTC parameters
                    logger.debug("Using RTC mode")
                    action_tensor = self._step_with_rtc(lerobot_obs)
                else:
                    # Use standard mode: predict_action_chunk and manage queue ourselves
                    # This avoids conflicts when sharing policy instance across sessions
                    logger.debug("Using standard mode")
                    action_tensor = self._step_without_rtc(lerobot_obs)
        
        # Apply postprocessor
        logger.debug("Applying postprocessor")
        action_tensor = self.postprocessor(action_tensor)
        
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

        logging.debug("One step completed")
        
        return poses_and_grippers

    def _step_without_rtc(self, lerobot_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform inference step without RTC (managing action queue ourselves).
        
        Args:
            lerobot_obs: Preprocessed LeRobot observation
            
        Returns:
            Single action tensor
        """
        # Check if we need to generate a new chunk
        if len(self._action_queue) == 0:
            t_inference_start = time.perf_counter()
            # Generate new action chunk
            action_chunk = self.policy.predict_action_chunk(lerobot_obs)
            t_inference_end = time.perf_counter()
            logger.debug(f"Inference time: {t_inference_end - t_inference_start:.2f}s")
            
            # Add actions to queue (only first n_action_step steps from chunk)
            # Shape: (batch_size, chunk_size, action_dim) -> extract timesteps
            if action_chunk.ndim == 3:
                chunk_size = action_chunk.shape[1]
                n_steps_to_add = min(self.n_action_step, chunk_size)
                for i in range(n_steps_to_add):
                    self._action_queue.append(action_chunk[:, i, :])
                logger.debug(f"Added {n_steps_to_add} action steps to queue (chunk size: {chunk_size})")
            else:
                # Single action case
                self._action_queue.append(action_chunk)
        
        # Pop next action from queue
        action = self._action_queue.popleft()
        return action

    def _step_with_rtc(self, lerobot_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform inference step with RTC enabled.
        
        Args:
            lerobot_obs: Preprocessed LeRobot observation
            
        Returns:
            Single action tensor
        """
        # Check if we need to generate a new chunk
        if len(self._action_queue) == 0:
            t_inference_start = time.perf_counter()
            
            # Get previous chunk leftover for RTC guidance
            prev_chunk_left_over = None
            if self._original_action_queue is not None:
                # Get remaining actions from previous chunk
                prev_chunk_left_over = self._original_action_queue
            
            # Calculate inference delay
            inference_delay = 0
            if self._inference_delays:
                # Use average of recent delays
                avg_delay = sum(self._inference_delays) / len(self._inference_delays)
                # Convert to timesteps (assuming ~10Hz, adjust if needed)
                inference_delay = max(1, math.ceil(avg_delay * 10))
            
            # Get execution horizon from config
            execution_horizon = None
            if (
                hasattr(self.policy, "config")
                and hasattr(self.policy.config, "rtc_config")
                and self.policy.config.rtc_config is not None
            ):
                execution_horizon = self.policy.config.rtc_config.execution_horizon
            
            # Generate new action chunk with RTC
            action_chunk = self.policy.predict_action_chunk(
                lerobot_obs,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
            )
            
            # Store original actions (before postprocessing) for next RTC iteration
            # Shape: (batch_size, chunk_size, action_dim) -> (chunk_size, action_dim)
            if action_chunk.ndim == 3:
                self._original_action_queue = action_chunk.squeeze(0).clone()
            else:
                self._original_action_queue = action_chunk.clone()
            
            # Track inference time
            # TODO: simulation inference delay is not the actual inference delay,
            # we need to align it with the simulation settings
            inference_time = time.perf_counter() - t_inference_start
            self._inference_delays.append(inference_time)
            
            # Add actions to queue (only first n_action_step steps from chunk)
            # Shape: (batch_size, chunk_size, action_dim) -> extract timesteps
            if action_chunk.ndim == 3:
                chunk_size = action_chunk.shape[1]
                n_steps_to_add = min(self.n_action_step, chunk_size)
                for i in range(n_steps_to_add):
                    self._action_queue.append(action_chunk[:, i, :])
                logger.debug(f"Added {n_steps_to_add} action steps to queue (chunk size: {chunk_size})")
            else:
                # Single action case
                self._action_queue.append(action_chunk)
        
        # Pop next action from queue
        action = self._action_queue.popleft()
        
        # Update original queue for next RTC iteration
        if self._original_action_queue is not None and len(self._original_action_queue) > 0:
            # Remove the first timestep that we just consumed
            self._original_action_queue = self._original_action_queue[1:]
        
        return action


class VLAPolicyBatch(Policy):
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
        policy_class: type[PreTrainedPolicy],
        checkpoint_path: str,
        device: torch.device,
        model_type: str,
        rtc_enabled_override: bool | None = None,
        compile_model_override: bool | None = None,
        task_instruction_override: str | None = None,
        n_action_step: int = 1,
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
            task_instruction_override: If provided, overrides the task instruction from observation.
                                     If None, uses observation.language_instruction.
            n_action_step: Number of action steps to add to queue from the front of action chunk
                          after each policy predict. Default is 1.
        """
        self.policy_class = policy_class
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_type = model_type
        self.rtc_enabled_override = rtc_enabled_override
        self.compile_model_override = compile_model_override
        self.task_instruction_override = task_instruction_override
        self.n_action_step = n_action_step
        
        # Load a SINGLE shared policy instance to save GPU memory
        # All sessions will share this policy instance for inference
        logger.info("Loading shared policy instance (will be shared across all sessions)")
        
        # Override compile_model config if specified (only for pi05 and pi0)
        config = None
        if compile_model_override is not None and model_type in ("pi05", "pi0"):
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(checkpoint_path)
            config.compile_model = compile_model_override
            logger.info(f"Overriding compile_model to {compile_model_override} for {model_type}")
        
        if config is not None:
            self.shared_policy = policy_class.from_pretrained(checkpoint_path, config=config)
        else:
            self.shared_policy = policy_class.from_pretrained(checkpoint_path)
        self.shared_policy.to(device)
        self.shared_policy.eval()

        logger.debug(f"Shared policy config: {self.shared_policy.config}")
        
        # Create preprocessor and postprocessor (also shared)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.shared_policy.config,
            pretrained_path=checkpoint_path,
        )
        
        # Internal UUID for non-batch interface
        self._internal_uuid = uuid.uuid4()
        
        # Mapping from UUID to VLAPolicy wrappers (each maintains its own state)
        # Note: All wrappers share the same underlying policy instance
        self._sub_policies: dict[uuid.UUID, VLAPolicy] = {}
        
        # Lock to protect shared policy access (prevent concurrent inference)
        # This ensures thread safety when multiple sessions access the shared policy
        self._policy_lock = threading.Lock()
        
        logger.info(
            f"Shared policy loaded. Model parameters: {sum(p.numel() for p in self.shared_policy.parameters()):,}"
        )

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
                vla_policy = VLAPolicy(
                    policy=self.shared_policy,  # Share the same policy instance
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    device=self.device,
                    rtc_enabled_override=self.rtc_enabled_override,
                    policy_lock=self._policy_lock,  # Pass the lock for thread safety
                    task_instruction_override=self.task_instruction_override,
                    n_action_step=self.n_action_step,
                )
                self._sub_policies[one_uuid] = vla_policy
            else:
                # Reset existing wrapper's state
                self._sub_policies[one_uuid].reset()

    def get_policy_metadata(self):
        """Get policy metadata."""
        return _get_policy_metadata(self.model_type, self.checkpoint_path)

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
        "--model-type",
        type=str,
        required=True,
        choices=["pi05", "groot", "smolvla"],
        help="VLA model type to use (pi05, groot, smolvla)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help="Enable RTC (Real-Time Chunking). If not specified, uses model config setting.",
    )
    parser.add_argument(
        "--disable-rtc",
        action="store_true",
        default=False,
        help="Disable RTC (Real-Time Chunking). Overrides --enable-rtc if both are specified.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for model optimization. Only applies to pi05 and pi0 models. "
             "If not specified, uses model config setting.",
    )
    parser.add_argument(
        "--no-compile-model",
        action="store_true",
        default=False,
        help="Disable torch.compile for model optimization. Only applies to pi05 and pi0 models. "
             "Overrides --compile-model if both are specified.",
    )
    parser.add_argument(
        "--task-instruction",
        type=str,
        default=None,
        help="Override task instruction from observation. If specified, this will replace "
             "observation.language_instruction for all inference steps.",
    )
    parser.add_argument(
        "--n-action-step",
        type=int,
        default=1,
        help="Number of action steps to add to queue from the front of action chunk "
             "after each policy predict. Default is 1. If set to a value larger than "
             "the chunk size, all steps in the chunk will be added.",
    )
    
    args = parser.parse_args()
    
    # Determine RTC setting: --disable-rtc takes precedence over --enable-rtc
    rtc_enabled_override = None
    if args.disable_rtc:
        rtc_enabled_override = False
        logger.info("RTC explicitly disabled via --disable-rtc")
    elif args.enable_rtc:
        rtc_enabled_override = True
        logger.info("RTC explicitly enabled via --enable-rtc")
    else:
        logger.info("RTC setting will be determined from model config")
    
    # Determine compile_model setting: --no-compile-model takes precedence over --compile-model
    compile_model_override = None
    if args.no_compile_model or args.compile_model:
        if args.model_type not in ("pi05", "pi0"):
            logger.warning(
                f"compile_model option is only supported for pi05 and pi0 models, "
                f"but model type is {args.model_type}. Ignoring compile_model option."
            )
        else:
            if args.no_compile_model:
                compile_model_override = False
                logger.info("compile_model explicitly disabled via --no-compile-model")
            elif args.compile_model:
                compile_model_override = True
                logger.info("compile_model explicitly enabled via --compile-model")
    else:
        logger.info("compile_model setting will be determined from model config")
    
    # Handle task instruction override
    task_instruction_override = args.task_instruction if args.task_instruction else None
    if task_instruction_override:
        logger.info(f"Task instruction override enabled: '{task_instruction_override}'")
    else:
        logger.info("Task instruction will be taken from observation.language_instruction")
    
    # Handle n_action_step
    n_action_step = args.n_action_step
    if n_action_step < 1:
        logger.warning(f"n_action_step must be >= 1, got {n_action_step}. Setting to 1.")
        n_action_step = 1
    logger.info(f"n_action_step set to {n_action_step} (will add first {n_action_step} steps from chunk to queue)")
    
    # Validate checkpoint path
    checkpoint_path = args.checkpoint_path
    if not Path(checkpoint_path).exists() and not checkpoint_path.startswith(("http://", "https://")):
        # Check if it's a HuggingFace model ID
        try:
            # Try to verify it's a valid HF model ID by checking if it contains /
            if "/" not in checkpoint_path:
                logger.warning(
                    f"Checkpoint path '{checkpoint_path}' doesn't exist locally. "
                    "Assuming it's a HuggingFace model ID."
                )
        except Exception:
            pass
    
    # Get device
    device = get_safe_torch_device(args.device, log=True)
    
    # Get policy class
    try:
        policy_class = get_policy_class(args.model_type)
    except ValueError as e:
        logger.error(f"Unsupported model type: {args.model_type}")
        logger.error(f"Supported types: pi05, groot, smolvla")
        raise
    
    logger.info(f"Loading {args.model_type} model from {checkpoint_path}")
    logger.info(f"Using device: {device}")
    
    # Create batch policy wrapper
    policy = VLAPolicyBatch(
        policy_class=policy_class,
        checkpoint_path=checkpoint_path,
        device=device,
        model_type=args.model_type,
        rtc_enabled_override=rtc_enabled_override,
        compile_model_override=compile_model_override,
        task_instruction_override=task_instruction_override,
        n_action_step=n_action_step,
    )
    
    logger.info("Policy wrapper initialized successfully")
    logger.info(f"Starting server on {args.server_uri}")
    
    # Start gRPC server
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()

