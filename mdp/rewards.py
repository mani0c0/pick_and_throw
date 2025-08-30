# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from .observations import object_grasped

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for lifting the object, masked by grasped state."""
    object: RigidObject = env.scene[object_cfg.name]
    lifted = (object.data.root_pos_w[:, 2] > minimal_height).float()
    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg).float()
    return lifted * grasped


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std)) * grasped


def object_velocity_goal_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    vel_command_name: str = "target_object_velocity",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    minimal_height: float = 0.1,
    ramp_at_vel: float = 0.5,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of the object's target linear velocity using an exponential kernel.

    The target velocity is assumed to be expressed in the robot base frame. It is rotated to
    the world frame and compared against the object's world linear velocity. The exponential
    kernel bounds the reward in [0, 1] and provides smooth gradients around zero, which is
    favorable for RL optimization.

    Args:
        env: The RL environment.
        std: The characteristic scale (in m/s). Smaller values emphasize tight tracking.
        command_name: Name of the command in the command manager providing (vx, vy, vz).
        object_cfg: Scene entity for the object whose velocity is tracked.
        robot_cfg: Scene entity for the robot providing the base frame orientation.
        ramp_at_vel: Magnitude (m/s) at which command scaling begins (> 0 keeps reward informative for small commands).
        ramp_rate: Linear rate to scale reward with command magnitude above ramp_at_vel.

    Returns:
        Tensor of shape (num_envs,) with values in (0, +inf) scaled by an exponential in [0, 1]
        and optionally multiplied by a command-magnitude factor >= 1.0.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_asset: RigidObject = env.scene[robot_cfg.name]

    target_vel_b = env.command_manager.get_command(vel_command_name)[:, :3]

    target_vel_w = quat_apply(robot_asset.data.root_quat_w, target_vel_b)

    current_vel_w = object_asset.data.root_lin_vel_w[:, :3]
    vel_error_sq = torch.sum(torch.square(target_vel_w - current_vel_w), dim=1)

    tracking_term = torch.exp(-vel_error_sq / (std * std))

    cmd_mag = torch.linalg.norm(target_vel_b, dim=1)
    cmd_scale = torch.clamp(1.0 + ramp_rate * (cmd_mag - ramp_at_vel), min=1.0)

    height_mask = (object_asset.data.root_pos_w[:, 2] > minimal_height)
    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)

    return tracking_term * cmd_scale * height_mask * grasped 


def object_pos_velocity_goal_reward(
    env: ManagerBasedRLEnv,
    std_vel_rel: float = 0.20,      # ~20% relative tolerance on velocity
    std_pos_abs: float = 0.20,      # absolute pos tolerance (meters)
    vel_command_name: str = "target_object_velocity",
    pos_command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    minimal_height: float = 0.04,
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_asset: RigidObject = env.scene[robot_cfg.name]

    target_vel_b = env.command_manager.get_command(vel_command_name)[:, :3]
    target_pos_b = env.command_manager.get_command(pos_command_name)[:, :3]

    target_vel_w = quat_apply(robot_asset.data.root_quat_w, target_vel_b)
    target_pos_w, _ = combine_frame_transforms(
        robot_asset.data.root_state_w[:, :3], robot_asset.data.root_state_w[:, 3:7], target_pos_b
    )

    vel_error = torch.linalg.norm(target_vel_w - object_asset.data.root_lin_vel_w[:, :3], dim=1)
    pos_error = torch.linalg.norm(target_pos_w - object_asset.data.root_pos_w[:, :3], dim=1)

    vel_mag = torch.linalg.norm(target_vel_b, dim=1)
    eps_v = 0.1  # m/s
    rel_vel_err = vel_error / (vel_mag + eps_v)

    vel_tracking = 1.0 - torch.tanh(rel_vel_err / (std_vel_rel + 1e-8))
    pos_tracking = 1.0 - torch.tanh(pos_error / (std_pos_abs + 1e-8))

    helper = 0.5 * vel_tracking + 0.2 * pos_tracking
    main = vel_tracking * pos_tracking

    height_mask = (object_asset.data.root_pos_w[:, 2] > minimal_height).float()
    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg).float()

    return (main + helper) * height_mask * grasped


def object_reached_velocity_goal_reward(
    env: ManagerBasedRLEnv,
    termination_name: str = "object_reached_velocity_goal",
) -> torch.Tensor:

    dones = env.termination_manager.get_term(termination_name)

    return dones


def grasp_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    object_asset: RigidObject = env.scene[object_cfg.name]
    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    height_mask = (object_asset.data.root_pos_w[:, 2] > minimal_height)

    return grasped * height_mask
