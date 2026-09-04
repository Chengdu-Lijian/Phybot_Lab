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
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = REPO_ROOT / "rsl_rl"
for path in (str(RSL_RL_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip
import numpy as np
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the txt file")
parser.add_argument("--fps", type=float, default=30.0, help="Target fps")
parser.add_argument("--loop_play", action="store_true", help="Loop motion playback until the app is closed.")
parser.add_argument(
    "--phybot_motion",
    type=str,
    default=None,
    help=(
        "Phybot motion file name or path. If a file name is given, it is resolved under "
        "legged_lab/envs/phybot_c2/datasets/motion_visualization. If suffix is omitted, .json is assumed."
    ),
)
parser.add_argument("--frame_start", type=int, default=0, help="Start frame index (inclusive) for phybot motion.")
parser.add_argument("--frame_end", type=int, default=None, help="End frame index (exclusive) for phybot motion.")
parser.add_argument(
    "--input_motion_path",
    type=str,
    default=None,
    help="Deprecated alias of --phybot_motion.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _save_frames_as_motion_json(save_path: str, frames: list[np.ndarray], fps: float):
    all_frames_np = np.stack(frames, axis=0)
    np.savetxt(save_path, all_frames_np, fmt="%f", delimiter=", ")

    with open(save_path, "r") as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(save_path, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = "  ["
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + "]\n")
            else:
                f.write(line_start_str + line.rstrip() + "],\n")

        f.write("]\n}")

    print(f"✅ Successfully converted to {save_path}")


def _resolve_joint_ids(env, joint_names: list[str]):
    joint_ids, _ = env.robot.find_joints(name_keys=joint_names, preserve_order=True)
    if len(joint_ids) == len(joint_names):
        return joint_ids

    fallback_names = [f"{name}_joint" for name in joint_names]
    joint_ids, _ = env.robot.find_joints(name_keys=fallback_names, preserve_order=True)
    if len(joint_ids) == len(joint_names):
        return joint_ids

    # Alternate naming used by legacy humanoid assets: hip_pitch_l_joint, etc.
    alt_map = {
        "left_hip_pitch": "hip_pitch_l_joint",
        "left_hip_roll": "hip_roll_l_joint",
        "left_hip_yaw": "hip_yaw_l_joint",
        "left_knee": "knee_pitch_l_joint",
        "left_ankle_pitch": "ankle_pitch_l_joint",
        "left_ankle_roll": "ankle_roll_l_joint",
        "right_hip_pitch": "hip_pitch_r_joint",
        "right_hip_roll": "hip_roll_r_joint",
        "right_hip_yaw": "hip_yaw_r_joint",
        "right_knee": "knee_pitch_r_joint",
        "right_ankle_pitch": "ankle_pitch_r_joint",
        "right_ankle_roll": "ankle_roll_r_joint",
        "left_shoulder_pitch": "shoulder_pitch_l_joint",
        "left_shoulder_roll": "shoulder_roll_l_joint",
        "left_shoulder_yaw": "shoulder_yaw_l_joint",
        "left_elbow_pitch": "elbow_pitch_l_joint",
        "right_shoulder_pitch": "shoulder_pitch_r_joint",
        "right_shoulder_roll": "shoulder_roll_r_joint",
        "right_shoulder_yaw": "shoulder_yaw_r_joint",
        "right_elbow_pitch": "elbow_pitch_r_joint",
    }
    alt_names = [alt_map.get(name, name) for name in joint_names]
    joint_ids, _ = env.robot.find_joints(name_keys=alt_names, preserve_order=True)
    if len(joint_ids) == len(joint_names):
        return joint_ids

    raise RuntimeError(
        "Failed to resolve all joint names for input motion playback. "
        f"Requested: {joint_names}. "
        "If you are using a phybot motion file, make sure to launch with --task=phybot_walk."
    )


def _resolve_phybot_motion_path(arg_value: str) -> Path:
    p = Path(arg_value).expanduser()
    if p.suffix == "":
        p = p.with_suffix(".json")
    if p.is_absolute() or p.parts[0] in {".", ".."} or "/" in arg_value:
        return p
    return Path("legged_lab/envs/phybot_c2/datasets/motion_visualization") / p


def play_amp_animation():
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    phybot_motion_arg = args_cli.phybot_motion or args_cli.input_motion_path
    if args_cli.loop_play and args_cli.save_path:
        raise ValueError("--loop_play cannot be used with --save_path (would create unbounded output).")

    all_frames = []
    if phybot_motion_arg is None:
        frame_cnt = 0
        while simulation_app.is_running():
            time = (frame_cnt % env.motion_len) * (1.0 / args_cli.fps)
            frame = env.visualize_motion(time)
            if args_cli.save_path:
                all_frames.append(frame.cpu().numpy().reshape(-1))
            frame_cnt += 1
            if (not args_cli.loop_play) and frame_cnt >= (env.motion_len - 1):
                break
    else:
        motion_path = _resolve_phybot_motion_path(phybot_motion_arg)
        with open(motion_path, "r") as f:
            input_motion = json.load(f)
        input_frames = np.asarray(input_motion["Frames"], dtype=np.float32)
        if input_frames.ndim != 2 or input_frames.shape[1] != 51:
            raise ValueError(
                f"Expected input motion with shape [N, 51], but got {input_frames.shape} "
                f"from {motion_path}"
            )
        frame_start = max(0, args_cli.frame_start)
        frame_end = input_frames.shape[0] if args_cli.frame_end is None else min(args_cli.frame_end, input_frames.shape[0])
        if frame_start >= frame_end:
            raise ValueError(
                f"Invalid frame range: frame_start={args_cli.frame_start}, frame_end={args_cli.frame_end}, "
                f"available=[0, {input_frames.shape[0]})"
            )
        input_frames = input_frames[frame_start:frame_end]
        print(f"Playing {motion_path} frames [{frame_start}, {frame_end}) with {input_frames.shape[0]} frames.")

        # Expected order from input AMP-obs style frames:
        # [base_lin(3), base_ang(3), gravity(3), joint_pos(21), joint_vel(21)]
        joint_order = [
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
        joint_ids = _resolve_joint_ids(env, joint_order)

        env_ids = torch.arange(env.num_envs, device=env.device)
        frame_idx = 0
        while simulation_app.is_running():
            row = torch.tensor(input_frames[frame_idx], device=env.device)
            q = row[9:30]
            dq = row[30:51]
            base_lin = row[0:3]
            base_ang = row[3:6]

            dof_pos = torch.zeros((env.num_envs, env.robot.num_joints), device=env.device)
            dof_vel = torch.zeros((env.num_envs, env.robot.num_joints), device=env.device)
            dof_pos[:, joint_ids] = q.unsqueeze(0).repeat(env.num_envs, 1)
            dof_vel[:, joint_ids] = dq.unsqueeze(0).repeat(env.num_envs, 1)
            env.robot.write_joint_position_to_sim(dof_pos)
            env.robot.write_joint_velocity_to_sim(dof_vel)

            # Keep the simulated root pose and only overwrite velocity.
            # Resetting pose every frame makes the base look pinned in space.
            root_state = env.robot.data.root_state_w.clone()
            root_state[:, 7:10] = base_lin.unsqueeze(0).repeat(env.num_envs, 1)
            root_state[:, 10:13] = base_ang.unsqueeze(0).repeat(env.num_envs, 1)
            env.robot.write_root_state_to_sim(root_state, env_ids)

            env.sim.render()
            env.sim.step()
            env.scene.update(dt=env.step_dt)

            if args_cli.save_path:
                amp_frame = env.get_amp_obs_for_expert_trans()
                all_frames.append(amp_frame[0].detach().cpu().numpy().reshape(-1))

            frame_idx += 1
            if frame_idx >= input_frames.shape[0]:
                if args_cli.loop_play:
                    frame_idx = 0
                else:
                    break

    if args_cli.save_path and all_frames:
        _save_frames_as_motion_json(args_cli.save_path, all_frames, args_cli.fps)
if __name__ == "__main__":
    play_amp_animation()
    simulation_app.close()
