# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class AutoOpenGripperPositionAction(BinaryJointPositionAction):
    """Binary gripper action that auto-opens on small tracking error.

    - Inherits the normal binary mapping (open/close) to joint position targets.
    - On each step, if both the object's position error relative to the commanded pose and
      the object's linear velocity error relative to the commanded velocity are below thresholds,
      the processed gripper action is overridden to the open command.
    """

    cfg: AutoOpenGripperPositionActionCfg

    def __init__(self, cfg: "AutoOpenGripperPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        # cache references
        self._object = self._env.scene[cfg.object_cfg.name]
        self._robot = self._env.scene[cfg.robot_cfg.name]

    def process_actions(self, actions: torch.Tensor):
        # first, process base binary action (sets self._processed_actions)
        super().process_actions(actions)

        mgr_env = cast("ManagerBasedRLEnv", self._env)

        # compute position error against commanded pose (expressed in robot base frame)
        target_pos_b = mgr_env.command_manager.get_command(self.cfg.pos_command_name)[:, :3]
        target_pos_w, _ = combine_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], target_pos_b
        )
        pos_err = torch.linalg.norm(target_pos_w - self._object.data.root_pos_w[:, :3], dim=1)

        # compute velocity error (rotate command from robot base to world and compare)
        target_vel_b = mgr_env.command_manager.get_command(self.cfg.vel_command_name)[:, :3]
        target_vel_w = quat_apply(self._robot.data.root_quat_w, target_vel_b)
        vel_err = torch.linalg.norm(target_vel_w - self._object.data.root_lin_vel_w[:, :3], dim=1)
        vel_mag = torch.linalg.norm(target_vel_b, dim=1)
        eps_v = 0.1  # m/s
        rel_vel_err = vel_err / (vel_mag + eps_v)

        # mask environments where both errors are under thresholds
        ok_mask = (pos_err < self.cfg.pos_error_threshold) & (rel_vel_err < self.cfg.vel_error_threshold)
        #ok_mask = rel_vel_err < self.cfg.vel_error_threshold
        if torch.any(ok_mask):
            # override processed action to open command for those envs
            self._processed_actions[ok_mask] = self._open_command


@configclass
class AutoOpenGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for AutoOpenGripperPositionAction.

    This wraps a binary joint position action for grippers and overrides the processed action
    to force the gripper to the open command when both the object's position and velocity errors
    are below configured thresholds.
    """

    class_type: type[ActionTerm] = AutoOpenGripperPositionAction

    # references
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    # command names to compute tracking errors against
    pos_command_name: str = "object_pose"
    vel_command_name: str = "target_object_velocity"

    # thresholds
    pos_error_threshold: float = 0.03  # meters
    vel_error_threshold: float = 0.15  # m/s
