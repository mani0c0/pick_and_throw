from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg_h import UniformVelocityCommandCfg_H


class UniformVelocityCommand_H(CommandTerm):
    """Command generator that generates a velocity command in SE(3) from uniform distribution."""
    cfg: UniformVelocityCommandCfg_H
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg_H, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, z_vel
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- linear velocity - z direction
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_vel_z)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.object.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xyz_velocity_to_arrow(self.command[:, :3])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xyz_velocity_to_arrow(self.object.data.root_lin_vel_b[:, :3])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xyz_velocity_to_arrow(self, xyz_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        marker_cfg: Any = self.goal_vel_visualizer.cfg.markers["arrow"]
        default_scale = getattr(marker_cfg, "scale")
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xyz_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xyz_velocity, dim=1) * 2
        # arrow-direction
        # rotate +X axis to align with the velocity direction in the base frame
        eps = 1.0e-8
        num_envs = xyz_velocity.shape[0]
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        # normalized direction (fallback to +X when velocity is near zero)
        vel_norm = torch.linalg.norm(xyz_velocity, dim=1, keepdim=True)
        dir_unit = torch.where(vel_norm > eps, xyz_velocity / vel_norm, x_axis)
        # compute rotation axis and angle between +X and dir_unit
        dot_prod = (x_axis * dir_unit).sum(dim=1).clamp(-1.0, 1.0)
        angle = torch.acos(dot_prod)
        axis = torch.cross(x_axis, dir_unit, dim=1)
        axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)
        # handle parallel/antiparallel cases
        # - if nearly parallel (angle ~ 0): identity
        # - if nearly antiparallel (angle ~ pi): rotate about +Y (any axis orthogonal to +X works)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(num_envs, 1)
        is_small_axis = axis_norm <= eps
        is_antiparallel = (~(vel_norm <= eps).squeeze(-1)) & (dot_prod < -1.0 + 1.0e-6)
        safe_axis = torch.where(is_antiparallel.unsqueeze(-1), y_axis, x_axis)
        axis = torch.where(is_small_axis, safe_axis, axis / axis_norm)
        # build quaternion in base frame, then convert to world
        arrow_quat = math_utils.quat_from_angle_axis(angle, axis)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
