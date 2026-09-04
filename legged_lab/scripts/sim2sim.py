# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# and is distributed under the BSD-3-Clause license.

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_viewer
import numpy as np
import torch

try:
    import glfw  # type: ignore
except Exception:
    glfw = None


@dataclass
class TaskPreset:
    model_rel_path: str
    num_action: int
    action_scale: float
    use_gait_phase: bool
    gait_air_ratio_l: float
    gait_air_ratio_r: float
    gait_phase_offset_l: float
    gait_phase_offset_r: float
    gait_cycle: float
    actor_obs_history_length: int
    clip_observations: float
    clip_actions: float
    dt: float
    decimation: int
    mujoco_joint_order: list[str]
    isaac_joint_order: list[str]
    default_dof_pos_by_joint: dict[str, float]
    control_mode: str = "position"
    soft_joint_pos_limit_factor: float = 1.0
    pd_kp_by_joint: dict[str, float] | None = None
    pd_kd_by_joint: dict[str, float] | None = None
    torque_limit_by_joint: dict[str, float] | None = None
    obs_scale_ang_vel: float = 1.0
    obs_scale_projected_gravity: float = 1.0
    obs_scale_commands: float = 1.0
    obs_scale_joint_pos: float = 1.0
    obs_scale_joint_vel: float = 1.0
    obs_scale_actions: float = 1.0
    cmd_range_x: tuple[float, float] | None = None
    cmd_range_y: tuple[float, float] | None = None
    cmd_range_yaw: tuple[float, float] | None = None


PHYBOT_JOINTS = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
]

# Extracted from Isaac runtime (`play.py` joint index print) for task `phybot_walk`.
PHYBOT_ISAAC_JOINTS = [
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_knee",
    "right_knee",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_elbow_pitch",
    "right_elbow_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
]

PHYBOT_PD_KP = {
    "left_hip_pitch": 100.0,
    "left_hip_roll": 100.0,
    "left_hip_yaw": 100.0,
    "left_knee": 100.0,
    "left_ankle_pitch": 50.0,
    "left_ankle_roll": 50.0,
    "right_hip_pitch": 100.0,
    "right_hip_roll": 100.0,
    "right_hip_yaw": 100.0,
    "right_knee": 100.0,
    "right_ankle_pitch": 50.0,
    "right_ankle_roll": 50.0,
    "waist_yaw": 100.0,
    "left_shoulder_pitch": 50.0,
    "left_shoulder_roll": 50.0,
    "left_shoulder_yaw": 5.0,
    "left_elbow_pitch": 50.0,
    "right_shoulder_pitch": 50.0,
    "right_shoulder_roll": 50.0,
    "right_shoulder_yaw": 5.0,
    "right_elbow_pitch": 50.0,
}

PHYBOT_PD_KD = {
    "left_hip_pitch": 10.0,
    "left_hip_roll": 10.0,
    "left_hip_yaw": 10.0,
    "left_knee": 10.0,
    "left_ankle_pitch": 5.0,
    "left_ankle_roll": 5.0,
    "right_hip_pitch": 10.0,
    "right_hip_roll": 10.0,
    "right_hip_yaw": 10.0,
    "right_knee": 10.0,
    "right_ankle_pitch": 5.0,
    "right_ankle_roll": 5.0,
    "waist_yaw": 10.0,
    "left_shoulder_pitch": 5.0,
    "left_shoulder_roll": 5.0,
    "left_shoulder_yaw": 5.0,
    "left_elbow_pitch": 5.0,
    "right_shoulder_pitch": 5.0,
    "right_shoulder_roll": 5.0,
    "right_shoulder_yaw": 5.0,
    "right_elbow_pitch": 5.0,
}

PHYBOT_TORQUE_LIMIT = {
    "left_hip_pitch": 200.0,
    "left_hip_roll": 80.0,
    "left_hip_yaw": 58.0,
    "left_knee": 80.0,
    "left_ankle_pitch": 58.0,
    "left_ankle_roll": 58.0,
    "right_hip_pitch": 200.0,
    "right_hip_roll": 80.0,
    "right_hip_yaw": 58.0,
    "right_knee": 80.0,
    "right_ankle_pitch": 58.0,
    "right_ankle_roll": 58.0,
    "waist_yaw": 58.0,
    "left_shoulder_pitch": 58.0,
    "left_shoulder_roll": 58.0,
    "left_shoulder_yaw": 58.0,
    "left_elbow_pitch": 9.0,
    "right_shoulder_pitch": 58.0,
    "right_shoulder_roll": 58.0,
    "right_shoulder_yaw": 58.0,
    "right_elbow_pitch": 58.0,
}

PRESETS: dict[str, TaskPreset] = {
    "phybot_walk": TaskPreset(
        model_rel_path="legged_lab/assets/phybot_c2/mjcf/phybot_c2.xml",
        num_action=21,
        action_scale=0.2,
        use_gait_phase=False,
        gait_air_ratio_l=0.38,
        gait_air_ratio_r=0.38,
        gait_phase_offset_l=0.38,
        gait_phase_offset_r=0.88,
        gait_cycle=0.85,
        actor_obs_history_length=10,
        clip_observations=100.0,
        clip_actions=100.0,
        dt=0.005,
        decimation=4,
        mujoco_joint_order=PHYBOT_JOINTS,
        isaac_joint_order=PHYBOT_ISAAC_JOINTS,
        control_mode="pd_torque",
        soft_joint_pos_limit_factor=0.9,
        pd_kp_by_joint=PHYBOT_PD_KP,
        pd_kd_by_joint=PHYBOT_PD_KD,
        torque_limit_by_joint=PHYBOT_TORQUE_LIMIT,
        obs_scale_ang_vel=0.25,
        obs_scale_projected_gravity=0.03,
        obs_scale_commands=1.0,
        obs_scale_joint_pos=1.0,
        obs_scale_joint_vel=0.05,
        obs_scale_actions=0.15,
        default_dof_pos_by_joint={
            "left_hip_pitch": -0.242,
            "left_hip_roll": 0.029,
            "left_hip_yaw": 0.007,
            "left_knee": 0.49,
            "left_ankle_pitch": -0.295,
            "left_ankle_roll": 0.0,
            "right_hip_pitch": -0.242,
            "right_hip_roll": -0.029,
            "right_hip_yaw": -0.007,
            "right_knee": 0.49,
            "right_ankle_pitch": -0.295,
            "right_ankle_roll": 0.0,
            "waist_yaw": 0.0,
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow_pitch": -0.17,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow_pitch": -0.17,
        },
        cmd_range_x=(-1.0, 3.0),
        cmd_range_y=(-0.2, 0.2),
        cmd_range_yaw=(-1.0, 1.0),
    ),
}


class SimToSimCfg:
    class sim:
        sim_duration = 100.0
        num_action = 20
        num_obs_per_step = 75
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        use_gait_phase = True
        gait_air_ratio_l = 0.38
        gait_air_ratio_r = 0.38
        gait_phase_offset_l = 0.38
        gait_phase_offset_r = 0.88
        gait_cycle = 0.85


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.
    """

    def __init__(
        self,
        cfg: SimToSimCfg,
        policy_path: str,
        model_path: str,
        preset: TaskPreset,
        headless: bool = False,
        debug_first_steps: int = 0,
        init_command_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        command_step: float = 0.2,
        command_limit: float = 1.0,
        respect_cmd_ranges: bool = True,
        reset_key: str = "backspace",
        camera_follow: bool = True,
        camera_follow_body: str = "base_link",
        camera_follow_smooth: float = 0.2,
        show_command_overlay: bool = True,
    ):
        self.cfg = cfg
        self.preset = preset
        self.headless = headless
        self.debug_first_steps = max(0, int(debug_first_steps))
        self.debug_step_count = 0
        self.init_command_vel = np.array(init_command_vel, dtype=np.float64)
        self.command_step = float(command_step)
        self.command_limit = float(command_limit)
        self.respect_cmd_ranges = bool(respect_cmd_ranges)
        self.reset_key = str(reset_key).lower()
        self.camera_follow = bool(camera_follow)
        self.camera_follow_body = str(camera_follow_body)
        self.camera_follow_smooth = float(camera_follow_smooth)
        self.show_command_overlay = bool(show_command_overlay)
        self.camera_follow_body_id = -1
        self._camera_lookat: np.ndarray | None = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self.viewer_key_hook_installed = False
        if not self.headless:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer._render_every_frame = False
            # Disable viewer pause/step hotkeys (Space/Right) to avoid conflicts with locomotion controls.
            self.viewer._paused = None
            self._configure_camera_follow()
            self._install_command_overlay_hook()
            self._install_viewer_key_hook()

        self.init_variables()
        self._validate_policy_dims()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        self.mujoco_joint_names = list(self.preset.mujoco_joint_order)
        self.isaac_joint_names = list(self.preset.isaac_joint_order)

        if len(self.mujoco_joint_names) != self.cfg.sim.num_action:
            raise ValueError("len(mujoco_joint_names) must match num_action")
        if len(self.isaac_joint_names) != self.cfg.sim.num_action:
            raise ValueError("len(isaac_joint_names) must match num_action")

        if set(self.mujoco_joint_names) != set(self.isaac_joint_names):
            raise ValueError("mujoco_joint_order and isaac_joint_order must contain the same joint names")

        self.mujoco_to_isaac_idx = np.array(
            [self.mujoco_joint_names.index(name) for name in self.isaac_joint_names], dtype=np.int64
        )
        self.isaac_to_mujoco_idx = np.array(
            [self.isaac_joint_names.index(name) for name in self.mujoco_joint_names], dtype=np.int64
        )

        self.qpos_addrs = np.zeros(self.cfg.sim.num_action, dtype=np.int64)
        self.qvel_addrs = np.zeros(self.cfg.sim.num_action, dtype=np.int64)
        self.joint_lower = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self.joint_upper = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        for i, joint_name in enumerate(self.mujoco_joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name!r} not found in model")
            self.qpos_addrs[i] = int(self.model.jnt_qposadr[joint_id])
            self.qvel_addrs[i] = int(self.model.jnt_dofadr[joint_id])

            # Build soft joint limits to match Isaac's soft_joint_pos_limit_factor behavior.
            if int(self.model.jnt_limited[joint_id]) == 1:
                low, high = self.model.jnt_range[joint_id]
            else:
                low, high = -np.inf, np.inf

            if np.isfinite(low) and np.isfinite(high):
                center = 0.5 * (low + high)
                half = 0.5 * (high - low) * float(self.preset.soft_joint_pos_limit_factor)
                self.joint_lower[i] = center - half
                self.joint_upper[i] = center + half
            else:
                self.joint_lower[i] = low
                self.joint_upper[i] = high

        self.default_dof_pos = np.array(
            [self.preset.default_dof_pos_by_joint[name] for name in self.mujoco_joint_names], dtype=np.float64
        )

        self.dof_pos = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self.dof_vel = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self.q_target = np.array(self.default_dof_pos, dtype=np.float64)

        if self.preset.control_mode == "pd_torque":
            if (
                self.preset.pd_kp_by_joint is None
                or self.preset.pd_kd_by_joint is None
                or self.preset.torque_limit_by_joint is None
            ):
                raise ValueError("pd_torque mode requires pd_kp_by_joint, pd_kd_by_joint, and torque_limit_by_joint.")
            self.pd_kp = np.array([self.preset.pd_kp_by_joint[name] for name in self.mujoco_joint_names], dtype=np.float64)
            self.pd_kd = np.array([self.preset.pd_kd_by_joint[name] for name in self.mujoco_joint_names], dtype=np.float64)
            self.torque_limit = np.array(
                [self.preset.torque_limit_by_joint[name] for name in self.mujoco_joint_names], dtype=np.float64
            )
        else:
            self.pd_kp = None
            self.pd_kd = None
            self.torque_limit = None

        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2, dtype=np.float64)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r], dtype=np.float64)
        self.phase_offset = np.array(
            [self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r], dtype=np.float64
        )

        self.command_vel = self.init_command_vel.copy()
        def _resolve_axis_range(axis_range: tuple[float, float] | None) -> tuple[float, float]:
            if not self.respect_cmd_ranges:
                return (-self.command_limit, self.command_limit)
            if axis_range is None:
                return (-self.command_limit, self.command_limit)
            lo, hi = float(axis_range[0]), float(axis_range[1])
            # Keep user limit as an extra safety envelope while respecting asymmetric training ranges.
            lo = max(lo, -self.command_limit)
            hi = min(hi, self.command_limit)
            if lo > hi:
                lo, hi = -self.command_limit, self.command_limit
            return (lo, hi)

        self.command_ranges = (
            _resolve_axis_range(self.preset.cmd_range_x),
            _resolve_axis_range(self.preset.cmd_range_y),
            _resolve_axis_range(self.preset.cmd_range_yaw),
        )
        for i, (lo, hi) in enumerate(self.command_ranges):
            self.command_vel[i] = np.clip(self.command_vel[i], lo, hi)
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )
        self.obs_history_initialized = False
        self.pending_reset = False

    def reset_simulation(self) -> None:
        """Reset simulator state and controller buffers to a clean start."""
        mujoco.mj_resetData(self.model, self.data)

        # Force controlled joints back to policy's default pose.
        self.data.qpos[self.qpos_addrs] = self.default_dof_pos
        self.data.qvel[self.qvel_addrs] = 0.0
        if self.preset.control_mode == "position":
            self.data.ctrl[:] = self.default_dof_pos
        else:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._read_joint_states()

        self.action.fill(0.0)
        self.q_target[:] = self.default_dof_pos
        self.obs_history.fill(0.0)
        self.obs_history_initialized = False
        self.command_vel[:] = self.init_command_vel
        for i, (lo, hi) in enumerate(self.command_ranges):
            self.command_vel[i] = np.clip(self.command_vel[i], lo, hi)
        self.episode_length_buf = 0
        self.gait_phase.fill(0.0)
        self.pending_reset = False
        self._camera_lookat = None
        x_lim, y_lim, yaw_lim = self.command_ranges
        print(
            f"[INFO] Simulation reset. cmd_range_x={x_lim}, cmd_range_y={y_lim}, cmd_range_yaw={yaw_lim}, "
            f"cmd_init=({self.command_vel[0]:+.2f}, {self.command_vel[1]:+.2f}, {self.command_vel[2]:+.2f})"
        )

    def _configure_camera_follow(self) -> None:
        if self.viewer is None or not self.camera_follow:
            return
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.camera_follow_body)
        if body_id < 0:
            # Fallback to first non-world body so follow still works on different robot naming schemes.
            if int(self.model.nbody) > 1:
                body_id = 1
                fallback_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"id={body_id}"
                print(
                    f"[WARN] camera_follow body not found: {self.camera_follow_body!r}. "
                    f"Fallback to root body: {fallback_name!r}."
                )
            else:
                print(f"[WARN] camera_follow body not found: {self.camera_follow_body!r}. Disable follow.")
                self.camera_follow = False
                return
        self.camera_follow_body_id = int(body_id)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def _update_camera_follow(self) -> None:
        if self.viewer is None or not self.camera_follow or self.camera_follow_body_id < 0:
            return
        target = np.array(self.data.xpos[self.camera_follow_body_id], dtype=np.float64)
        alpha = float(np.clip(self.camera_follow_smooth, 0.0, 1.0))
        if self._camera_lookat is None:
            self._camera_lookat = target
        elif alpha <= 0.0:
            self._camera_lookat = target
        else:
            self._camera_lookat = (1.0 - alpha) * self._camera_lookat + alpha * target
        self.viewer.cam.lookat[:] = self._camera_lookat

    def _install_command_overlay_hook(self) -> None:
        if self.viewer is None or not self.show_command_overlay:
            return
        original_create_overlay = self.viewer._create_overlay

        def wrapped_create_overlay():
            original_create_overlay()
            cmd = self.command_vel
            grid = mujoco.mjtGridPos.mjGRID_TOPLEFT
            if grid not in self.viewer._overlay:
                self.viewer._overlay[grid] = ["", ""]
            self.viewer._overlay[grid][0] += "Policy command [x y yaw]\n"
            self.viewer._overlay[grid][1] += f"{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}\n"

        self.viewer._create_overlay = wrapped_create_overlay

    def _install_viewer_key_hook(self) -> None:
        """Install keyboard handling on MuJoCo window callback to avoid pynput dropouts."""
        if self.viewer is None:
            return
        if glfw is None:
            print("[WARN] glfw module unavailable; cannot install viewer key hook.")
            return
        original_key_callback = self.viewer._key_callback

        def wrapped_key_callback(window, key, scancode, action, mods):
            # Handle locomotion command keys on key-release to match viewer behavior.
            if action == glfw.RELEASE:
                if self._handle_viewer_key_event(key):
                    return
            original_key_callback(window, key, scancode, action, mods)

        self.viewer._key_callback = wrapped_key_callback
        glfw.set_key_callback(self.viewer.window, wrapped_key_callback)
        self.viewer_key_hook_installed = True

    def _handle_viewer_key_event(self, key: int) -> bool:
        """Handle key events from MuJoCo/GLFW callback. Returns True if consumed."""
        if glfw is None:
            return False
        key_delete = getattr(glfw, "KEY_DELETE", -9999)
        key_kp_7 = getattr(glfw, "KEY_KP_7", -9999)
        key_kp_8 = getattr(glfw, "KEY_KP_8", -9999)

        if self.reset_key == "backspace" and key in (glfw.KEY_BACKSPACE, key_delete):
            self.pending_reset = True
            print("[INFO] Reset requested.")
            return True
        if self.reset_key == "x" and key == glfw.KEY_X:
            self.pending_reset = True
            print("[INFO] Reset requested.")
            return True
        if self.reset_key == "r" and key == glfw.KEY_R:
            self.pending_reset = True
            print("[INFO] Reset requested.")
            return True

        if key == glfw.KEY_UP:
            self.adjust_command_vel(0, self.command_step)
            return True
        if key == glfw.KEY_DOWN:
            self.adjust_command_vel(0, -self.command_step)
            return True
        if key == glfw.KEY_LEFT:
            self.adjust_command_vel(1, -self.command_step)
            return True
        if key == glfw.KEY_RIGHT:
            self.adjust_command_vel(1, self.command_step)
            return True
        if key in (glfw.KEY_7, key_kp_7, glfw.KEY_Q):
            self.adjust_command_vel(2, -self.command_step)
            return True
        if key in (glfw.KEY_8, key_kp_8, glfw.KEY_E):
            self.adjust_command_vel(2, self.command_step)
            return True
        if key == glfw.KEY_X:
            self.command_vel[:] = 0.0
            for i, (lo, hi) in enumerate(self.command_ranges):
                self.command_vel[i] = np.clip(self.command_vel[i], lo, hi)
            print(f"[CMD] x={self.command_vel[0]:+.2f}, y={self.command_vel[1]:+.2f}, yaw={self.command_vel[2]:+.2f}")
            print("[INFO] Command velocity cleared.")
            return True

        return False

    def _infer_policy_input_dim(self) -> int | None:
        for _, param in self.policy.state_dict().items():
            if getattr(param, "ndim", 0) == 2:
                return int(param.shape[1])
        return None

    def _infer_policy_output_dim(self) -> int | None:
        linear_weights = [p for p in self.policy.state_dict().values() if getattr(p, "ndim", 0) == 2]
        if not linear_weights:
            return None
        return int(linear_weights[-1].shape[0])

    def _validate_policy_dims(self) -> None:
        expected_obs_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        expected_action_dim = self.cfg.sim.num_action
        policy_obs_dim = self._infer_policy_input_dim()
        policy_action_dim = self._infer_policy_output_dim()

        if policy_obs_dim is not None and policy_obs_dim != expected_obs_dim:
            raise ValueError(
                f"Policy obs dim mismatch: policy={policy_obs_dim}, expected={expected_obs_dim}. "
                "Check task/preset or gait setting."
            )
        if policy_action_dim is not None and policy_action_dim != expected_action_dim:
            raise ValueError(
                f"Policy action dim mismatch: policy={policy_action_dim}, expected={expected_action_dim}. "
                "Check task/preset."
            )

    def _read_joint_states(self) -> None:
        self.dof_pos[:] = self.data.qpos[self.qpos_addrs]
        self.dof_vel[:] = self.data.qvel[self.qvel_addrs]

    def get_obs(self) -> np.ndarray:
        """Compute current observation vector from MuJoCo sensors and internal state."""
        self._read_joint_states()

        obs_items = [
            self.data.sensor("angular-velocity").data.astype(np.float64) * self.preset.obs_scale_ang_vel,  # 3
            self.quat_rotate_inverse(
                self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64), np.array([0.0, 0.0, -1.0])
            )
            * self.preset.obs_scale_projected_gravity,  # 3
            self.command_vel * self.preset.obs_scale_commands,  # 3
            (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx] * self.preset.obs_scale_joint_pos,
            self.dof_vel[self.mujoco_to_isaac_idx] * self.preset.obs_scale_joint_vel,
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions) * self.preset.obs_scale_actions,
        ]

        if self.cfg.robot.use_gait_phase:
            obs_items.extend(
                [
                    np.sin(2.0 * np.pi * self.gait_phase),
                    np.cos(2.0 * np.pi * self.gait_phase),
                    self.phase_ratio,
                ]
            )

        obs = np.concatenate(obs_items, axis=0).astype(np.float32)

        if not self.obs_history_initialized:
            self.obs_history[:] = np.tile(obs, self.cfg.sim.actor_obs_history_length)
            self.obs_history_initialized = True
        else:
            self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
            self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def _compute_joint_position_target(self) -> np.ndarray:
        """Compute joint position targets (MuJoCo joint order), then clamp to soft limits."""
        actions_scaled = self.action * self.cfg.sim.action_scale
        q_target = actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos
        q_target = np.clip(q_target, self.joint_lower, self.joint_upper)
        self.q_target[:] = q_target
        return q_target

    def compute_control(self) -> np.ndarray:
        """
        Compute actuator control command.
        - position: feed desired joint positions directly (for position actuators).
        - pd_torque: convert desired joint positions to torques for motor actuators.
        """
        q_target = self._compute_joint_position_target()

        if self.preset.control_mode == "position":
            return q_target
        if self.preset.control_mode == "pd_torque":
            assert self.pd_kp is not None and self.pd_kd is not None and self.torque_limit is not None
            tau = self.pd_kp * (q_target - self.dof_pos) - self.pd_kd * self.dof_vel
            return np.clip(tau, -self.torque_limit, self.torque_limit)
        raise ValueError(f"Unsupported control mode: {self.preset.control_mode}")

    def _debug_print_step(self) -> None:
        if self.debug_step_count >= self.debug_first_steps:
            return
        at_lower = np.isclose(self.q_target, self.joint_lower, atol=1e-4)
        at_upper = np.isclose(self.q_target, self.joint_upper, atol=1e-4)
        saturated_idx = np.where(at_lower | at_upper)[0].tolist()
        saturated_names = [self.mujoco_joint_names[i] for i in saturated_idx]
        print(
            f"[DEBUG step={self.debug_step_count}] "
            f"cmd={self.command_vel.tolist()} "
            f"action_minmax=({float(self.action.min()):.3f}, {float(self.action.max()):.3f}) "
            f"q_target_minmax=({float(self.q_target.min()):.3f}, {float(self.q_target.max()):.3f}) "
            f"saturated_joints={saturated_names}"
        )
        self.debug_step_count += 1

    def run(self) -> None:
        """Run the simulation loop with keyboard-controlled commands."""
        self.reset_simulation()
        self.setup_keyboard_listener()
        if self.listener is not None:
            self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            if self.pending_reset:
                self.reset_simulation()
            if self.listener is not None and not self.listener.is_alive():
                print("[WARN] Keyboard listener stopped unexpectedly. Restarting listener.")
                self.setup_keyboard_listener()
                if self.listener is not None:
                    self.listener.start()

            self.obs_history = self.get_obs()
            action = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()
            self.action[:] = action[: self.cfg.sim.num_action]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            for _ in range(self.cfg.sim.decimation):
                step_start_time = time.time()
                self._read_joint_states()
                self.data.ctrl = self.compute_control()
                self._debug_print_step()
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self._update_camera_follow()
                    self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            self.episode_length_buf += 1
            self.calculate_gait_para()

        if self.listener is not None:
            self.listener.stop()
        if self.viewer is not None:
            self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by the inverse of a quaternion."""
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self) -> None:
        """Update gait phase parameters based on simulation time and offset."""
        if not self.cfg.robot.use_gait_phase:
            return
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """Adjust command velocity vector."""
        self.command_vel[idx] += increment
        lo, hi = self.command_ranges[idx]
        self.command_vel[idx] = np.clip(self.command_vel[idx], lo, hi)
        print(f"[CMD] x={self.command_vel[0]:+.2f}, y={self.command_vel[1]:+.2f}, yaw={self.command_vel[2]:+.2f}")

    def setup_keyboard_listener(self) -> None:
        """Set up keyboard event listener for user control input."""
        self.listener = None
        if self.headless:
            return
        if self.viewer_key_hook_installed:
            print(
                "[INFO] Keyboard map (viewer callback): "
                "x+/x-=UP/DOWN, y+/y-=RIGHT/LEFT, yaw+/yaw-=8/7 or E/Q, clear=x, reset="
                f"{self.reset_key} (backspace accepts Delete too)."
            )
            return
        try:
            from pynput import keyboard
        except Exception as exc:
            print(f"[WARN] Keyboard control disabled (pynput unavailable): {exc}")
            return

        def on_press(key):
            try:
                key_char = getattr(key, "char", None)

                is_reset = False
                if self.reset_key == "backspace":
                    is_reset = key in (keyboard.Key.backspace, keyboard.Key.delete)
                elif self.reset_key == "x":
                    is_reset = key_char in ("x", "X")
                elif self.reset_key == "r":
                    is_reset = key_char in ("r", "R")

                if is_reset:
                    self.pending_reset = True
                    print("[INFO] Reset requested.")
                    return

                if key == keyboard.Key.up:
                    self.adjust_command_vel(0, self.command_step)
                elif key == keyboard.Key.down:
                    self.adjust_command_vel(0, -self.command_step)
                elif key == keyboard.Key.left:
                    self.adjust_command_vel(1, -self.command_step)
                elif key == keyboard.Key.right:
                    self.adjust_command_vel(1, self.command_step)
                elif key_char in ("7", "q", "Q"):
                    self.adjust_command_vel(2, -self.command_step)
                elif key_char in ("8", "e", "E"):
                    self.adjust_command_vel(2, self.command_step)
                elif key_char in ("x", "X"):
                    self.command_vel[:] = 0.0
                    for i, (lo, hi) in enumerate(self.command_ranges):
                        self.command_vel[i] = np.clip(self.command_vel[i], lo, hi)
                    print(
                        f"[CMD] x={self.command_vel[0]:+.2f}, y={self.command_vel[1]:+.2f}, yaw={self.command_vel[2]:+.2f}"
                    )
                    print("[INFO] Command velocity cleared.")
            except Exception as exc:
                print(f"[WARN] keyboard callback error: {exc}")

        self.listener = keyboard.Listener(on_press=on_press)
        print(
            "[INFO] Keyboard map (pynput fallback): "
            "x+/x-=UP/DOWN, y+/y-=RIGHT/LEFT, yaw+/yaw-=8/7 or E/Q, clear=x, reset="
            f"{self.reset_key}. "
            "Note: keys 0-5 are reserved by MuJoCo viewer (geomgroup visibility toggles)."
        )


def _find_policy_in_exported_dir(exported_dir: Path) -> Path | None:
    if not exported_dir.is_dir():
        return None

    preferred = exported_dir / "policy.pt"
    if preferred.is_file():
        return preferred

    pt_files = sorted(p for p in exported_dir.iterdir() if p.is_file() and p.suffix == ".pt")
    if pt_files:
        return pt_files[0]

    return None


def _latest_task_policy(task: str, root: Path) -> str | None:
    base = root / "logs" / task
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    for run in reversed(runs):
        candidate = _find_policy_in_exported_dir(run / "exported")
        if candidate is not None:
            return str(candidate)
    return None


def _legacy_policy_candidates(task: str, root: Path) -> list[Path]:
    export_dir = root / "Exported_policy"
    names = [f"{task}.pt"]
    if task.startswith("phybot_"):
        names.append(f"{task.removeprefix('phybot_')}.pt")
    return [export_dir / name for name in names]


def _resolve_policy_path(task: str, root: Path, requested_policy: str | None) -> str:
    latest = _latest_task_policy(task, root)

    if requested_policy is not None:
        requested = Path(requested_policy).expanduser()
        if not requested.is_absolute():
            requested = root / requested
        requested = requested.resolve(strict=False)

        if requested.is_file():
            return str(requested)

        if requested.is_dir():
            candidate = _find_policy_in_exported_dir(requested)
            if candidate is not None:
                print(f"[INFO] Resolved policy directory to file: {candidate}")
                return str(candidate)
            print(f"[WARN] No .pt policy file found in directory: {requested}")
        else:
            print(f"[WARN] Requested policy path does not exist: {requested}")

        if latest is not None:
            print(f"[INFO] Falling back to latest exported policy: {latest}")
            return latest

        for candidate in _legacy_policy_candidates(task, root):
            if candidate.is_file():
                print(f"[INFO] Falling back to legacy exported policy: {candidate}")
                return str(candidate)

        return str(requested)

    if latest is not None:
        return latest

    for candidate in _legacy_policy_candidates(task, root):
        if candidate.is_file():
            return str(candidate)

    return str(_legacy_policy_candidates(task, root)[0])


def _load_training_overrides_from_policy(preset: TaskPreset, policy_path: str) -> TaskPreset:
    """
    If policy path is from logs/<exp>/<run>/exported/policy.pt, load params/env.yaml
    and override runtime preset fields to match training-time settings.
    """
    policy_file = Path(policy_path).expanduser().resolve()
    run_dir = policy_file.parent.parent
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return preset

    try:
        import yaml
    except Exception:
        print("[WARN] PyYAML not available; skip loading training-time env overrides.")
        return preset

    try:
        with env_yaml.open("r", encoding="utf-8") as f:
            env_cfg = yaml.unsafe_load(f)
    except Exception as exc:
        print(f"[WARN] Failed to parse {env_yaml}: {exc}")
        return preset

    out = copy.deepcopy(preset)
    try:
        robot_cfg = env_cfg.get("robot", {})
        norm_cfg = env_cfg.get("normalization", {})
        gait_cfg = env_cfg.get("gait", {})
        commands_cfg = env_cfg.get("commands", {})
        obs_scales = norm_cfg.get("obs_scales", {})
        init_joint_pos = env_cfg.get("scene", {}).get("robot", {}).get("init_state", {}).get("joint_pos", {})

        if "action_scale" in robot_cfg:
            out.action_scale = float(robot_cfg["action_scale"])
        if "actor_obs_history_length" in robot_cfg:
            out.actor_obs_history_length = int(robot_cfg["actor_obs_history_length"])
        if "enable" in gait_cfg:
            out.use_gait_phase = bool(gait_cfg["enable"])
        if "gait_air_ratio_l" in gait_cfg:
            out.gait_air_ratio_l = float(gait_cfg["gait_air_ratio_l"])
        if "gait_air_ratio_r" in gait_cfg:
            out.gait_air_ratio_r = float(gait_cfg["gait_air_ratio_r"])
        if "gait_phase_offset_l" in gait_cfg:
            out.gait_phase_offset_l = float(gait_cfg["gait_phase_offset_l"])
        if "gait_phase_offset_r" in gait_cfg:
            out.gait_phase_offset_r = float(gait_cfg["gait_phase_offset_r"])
        if "gait_cycle" in gait_cfg:
            out.gait_cycle = float(gait_cfg["gait_cycle"])

        if "ang_vel" in obs_scales:
            out.obs_scale_ang_vel = float(obs_scales["ang_vel"])
        if "projected_gravity" in obs_scales:
            out.obs_scale_projected_gravity = float(obs_scales["projected_gravity"])
        if "commands" in obs_scales:
            out.obs_scale_commands = float(obs_scales["commands"])
        if "joint_pos" in obs_scales:
            out.obs_scale_joint_pos = float(obs_scales["joint_pos"])
        if "joint_vel" in obs_scales:
            out.obs_scale_joint_vel = float(obs_scales["joint_vel"])
        if "actions" in obs_scales:
            out.obs_scale_actions = float(obs_scales["actions"])

        ranges_cfg = commands_cfg.get("ranges", {}) if isinstance(commands_cfg, dict) else {}
        if "lin_vel_x" in ranges_cfg and ranges_cfg["lin_vel_x"] is not None:
            out.cmd_range_x = (float(ranges_cfg["lin_vel_x"][0]), float(ranges_cfg["lin_vel_x"][1]))
        if "lin_vel_y" in ranges_cfg and ranges_cfg["lin_vel_y"] is not None:
            out.cmd_range_y = (float(ranges_cfg["lin_vel_y"][0]), float(ranges_cfg["lin_vel_y"][1]))
        if "ang_vel_z" in ranges_cfg and ranges_cfg["ang_vel_z"] is not None:
            out.cmd_range_yaw = (float(ranges_cfg["ang_vel_z"][0]), float(ranges_cfg["ang_vel_z"][1]))

        # Keep joint ordering from preset but refresh default pose values if available.
        if isinstance(init_joint_pos, dict) and len(init_joint_pos) > 0:
            for name in out.default_dof_pos_by_joint.keys():
                if name in init_joint_pos:
                    out.default_dof_pos_by_joint[name] = float(init_joint_pos[name])

        print(f"[INFO] Loaded runtime overrides from training config: {env_yaml}")
    except Exception as exc:
        print(f"[WARN] Failed applying overrides from {env_yaml}: {exc}")
        return preset

    return out


def _prepare_model_path(model_path: str) -> str:
    """
    Resolve known asset-path issues in MuJoCo XMLs without modifying source files.

    Current fix:
    - phybot_c2.xml uses meshdir="../new_meshes/" in some branches, while this repo stores meshes in "../meshes/".
    """
    model_file = Path(model_path)
    if not model_file.exists():
        return model_path

    try:
        xml_text = model_file.read_text(encoding="utf-8")
    except Exception:
        return model_path

    meshdir_old = 'meshdir="../new_meshes/"'
    meshdir_new = 'meshdir="../meshes/"'
    if meshdir_old not in xml_text:
        return model_path

    old_dir = (model_file.parent / "../new_meshes").resolve()
    new_dir = (model_file.parent / "../meshes").resolve()
    if old_dir.exists() or not new_dir.exists():
        return model_path

    abs_meshdir = new_dir.as_posix().rstrip("/") + "/"
    patched_text = xml_text.replace(meshdir_old, f'meshdir="{abs_meshdir}"')
    patched_path = Path("/tmp") / f"{model_file.stem}_meshdir_fixed.xml"
    patched_path.write_text(patched_text, encoding="utf-8")
    print(
        f"[WARN] Auto-fixed meshdir in temporary model: {patched_path} "
        f"(original referenced missing directory: {old_dir})"
    )
    return str(patched_path)


def build_cfg_from_preset(preset: TaskPreset, duration: float) -> SimToSimCfg:
    cfg = SimToSimCfg()
    cfg.sim.sim_duration = duration
    cfg.sim.num_action = preset.num_action
    cfg.sim.dt = preset.dt
    cfg.sim.decimation = preset.decimation
    cfg.sim.clip_observations = preset.clip_observations
    cfg.sim.clip_actions = preset.clip_actions
    cfg.sim.action_scale = preset.action_scale
    cfg.sim.actor_obs_history_length = preset.actor_obs_history_length
    cfg.robot.use_gait_phase = preset.use_gait_phase
    cfg.robot.gait_air_ratio_l = preset.gait_air_ratio_l
    cfg.robot.gait_air_ratio_r = preset.gait_air_ratio_r
    cfg.robot.gait_phase_offset_l = preset.gait_phase_offset_l
    cfg.robot.gait_phase_offset_r = preset.gait_phase_offset_r
    cfg.robot.gait_cycle = preset.gait_cycle

    cfg.sim.num_obs_per_step = 9 + 3 * cfg.sim.num_action + (6 if cfg.robot.use_gait_phase else 0)
    return cfg


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="phybot_walk",
        choices=["phybot_walk"],
        help="Task preset: phybot_walk (Phybot C2)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy file or exported directory. Missing/omitted values fall back to the latest exported task policy.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to MuJoCo XML model. If omitted, choose task-specific default.",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--debug_first_steps",
        type=int,
        default=0,
        help="Print action/target saturation diagnostics for the first N control steps",
    )
    parser.add_argument("--cmd_x", type=float, default=0.0, help="Initial command velocity x")
    parser.add_argument("--cmd_y", type=float, default=0.0, help="Initial command velocity y")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Initial command yaw")
    parser.add_argument("--cmd_step", type=float, default=0.2, help="Keyboard command increment step")
    parser.add_argument("--cmd_limit", type=float, default=3.0, help="Absolute clamp for command velocity")
    parser.add_argument(
        "--respect_cmd_ranges",
        dest="respect_cmd_ranges",
        action="store_true",
        help="Clamp each command axis to training-time ranges from params/env.yaml (default: enabled)",
    )
    parser.add_argument(
        "--ignore_cmd_ranges",
        dest="respect_cmd_ranges",
        action="store_false",
        help="Ignore training-time per-axis ranges and use only --cmd_limit for all axes",
    )
    parser.set_defaults(respect_cmd_ranges=True)
    parser.add_argument(
        "--reset_key",
        type=str,
        default="backspace",
        choices=["backspace", "x", "r"],
        help="Reset hotkey. Default 'backspace' avoids conflict with MuJoCo viewer 'r' transparency toggle.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without MuJoCo viewer (useful for quick validation/smoke tests)",
    )
    parser.add_argument(
        "--camera_follow",
        dest="camera_follow",
        action="store_true",
        help="Enable camera follow on a target body (default: enabled)",
    )
    parser.add_argument(
        "--no_camera_follow",
        dest="camera_follow",
        action="store_false",
        help="Disable camera follow and keep manual/free camera",
    )
    parser.set_defaults(camera_follow=True)
    parser.add_argument(
        "--camera_follow_body",
        type=str,
        default="base_link",
        help="MuJoCo body name used as camera follow target",
    )
    parser.add_argument(
        "--camera_follow_smooth",
        type=float,
        default=0.2,
        help="Smoothing factor for camera lookat update in [0,1] (0 means no smoothing)",
    )
    parser.add_argument(
        "--show_command_overlay",
        dest="show_command_overlay",
        action="store_true",
        help="Show current policy command [x, y, yaw] in viewer top-left overlay (default: enabled)",
    )
    parser.add_argument(
        "--no_show_command_overlay",
        dest="show_command_overlay",
        action="store_false",
        help="Hide current policy command in viewer overlay",
    )
    parser.set_defaults(show_command_overlay=True)
    args = parser.parse_args()

    preset = PRESETS[args.task]
    policy_path = _resolve_policy_path(args.task, root_dir, args.policy)
    preset = _load_training_overrides_from_policy(preset, policy_path)
    model_path = args.model or str(root_dir / preset.model_rel_path)
    model_path = _prepare_model_path(model_path)

    if not os.path.isfile(policy_path):
        print(f"[ERROR] Policy file not found: {policy_path}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"[ERROR] MuJoCo model file not found: {model_path}")
        sys.exit(1)

    cfg = build_cfg_from_preset(preset, args.duration)

    print(f"[INFO] Loaded task preset: {args.task}")
    print(f"[INFO] Loaded policy: {policy_path}")
    print(f"[INFO] Loaded model: {model_path}")
    print(
        f"[INFO] num_action={cfg.sim.num_action}, num_obs_per_step={cfg.sim.num_obs_per_step}, "
        f"history={cfg.sim.actor_obs_history_length}, total_obs={cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length}"
    )
    print(
        "[INFO] runtime alignment: "
        f"action_scale={preset.action_scale}, "
        f"obs_scales={{ang_vel:{preset.obs_scale_ang_vel}, gravity:{preset.obs_scale_projected_gravity}, "
        f"cmd:{preset.obs_scale_commands}, joint_pos:{preset.obs_scale_joint_pos}, "
        f"joint_vel:{preset.obs_scale_joint_vel}, prev_action:{preset.obs_scale_actions}}}, "
        f"use_gait_phase={preset.use_gait_phase}"
    )
    print(
        f"[INFO] command control: init=({args.cmd_x}, {args.cmd_y}, {args.cmd_yaw}), "
        f"step={args.cmd_step}, limit={args.cmd_limit}, respect_cmd_ranges={args.respect_cmd_ranges}, "
        f"reset_key={args.reset_key}"
    )
    print(
        "[INFO] command training-ranges: "
        f"x={preset.cmd_range_x}, y={preset.cmd_range_y}, yaw={preset.cmd_range_yaw}"
    )
    print(
        f"[INFO] viewer: camera_follow={args.camera_follow}, "
        f"camera_follow_body={args.camera_follow_body}, camera_follow_smooth={args.camera_follow_smooth}, "
        f"show_command_overlay={args.show_command_overlay}"
    )

    runner = MujocoRunner(
        cfg=cfg,
        policy_path=policy_path,
        model_path=model_path,
        preset=preset,
        headless=args.headless,
        debug_first_steps=args.debug_first_steps,
        init_command_vel=(args.cmd_x, args.cmd_y, args.cmd_yaw),
        command_step=args.cmd_step,
        command_limit=args.cmd_limit,
        respect_cmd_ranges=args.respect_cmd_ranges,
        reset_key=args.reset_key,
        camera_follow=args.camera_follow,
        camera_follow_body=args.camera_follow_body,
        camera_follow_smooth=args.camera_follow_smooth,
        show_command_overlay=args.show_command_overlay,
    )
    runner.run()
