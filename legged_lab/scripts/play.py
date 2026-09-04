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
import os
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = REPO_ROOT / "rsl_rl"
for path in (str(RSL_RL_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

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

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.envs.phybot_c2.symmetry import validate_symmetry_involution
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _print_policy_obs_layout(env, obs: torch.Tensor):
    num_actions = int(env.num_actions)
    history_len = int(env.cfg.robot.actor_obs_history_length)
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))
    per_step_fields = [
        ("root_ang_vel_b", 3),
        ("projected_gravity_b", 3),
        ("command", 3),
        ("joint_pos_delta_from_default", num_actions),
        ("joint_vel_delta_from_default", num_actions),
        ("previous_action", num_actions),
    ]
    if use_gait_phase:
        per_step_fields.append(("gait_phase_features", 6))

    per_step_dim = sum(dim for _, dim in per_step_fields)
    core_dim = per_step_dim * history_len
    total_dim = int(obs.shape[-1])
    tail_dim = max(0, total_dim - core_dim)

    print(f"[OBS] policy obs total dim = {total_dim}")
    print(f"[OBS] history_len = {history_len}, per_step_dim = {per_step_dim}, core_dim = {core_dim}, tail_dim = {tail_dim}")
    for name, dim in per_step_fields:
        print(f"[OBS] per-step field: {name:<30} dim={dim}")
    if tail_dim > 0:
        print(f"[OBS] tail field: height_scan_or_extra_features dim={tail_dim}")


def _print_obs_grouped(env, obs: torch.Tensor, step_count: int, env_id: int = 0):
    num_actions = int(env.num_actions)
    history_len = int(env.cfg.robot.actor_obs_history_length)
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))
    per_step_fields = [
        ("root_ang_vel_b", 3),
        ("projected_gravity_b", 3),
        ("command", 3),
        ("joint_pos_delta_from_default", num_actions),
        ("joint_vel_delta_from_default", num_actions),
        ("previous_action", num_actions),
    ]
    if use_gait_phase:
        per_step_fields.append(("gait_phase_features", 6))

    obs_env = obs[env_id].detach().cpu().flatten()
    total_dim = int(obs_env.numel())
    per_step_dim = sum(dim for _, dim in per_step_fields)
    core_dim = per_step_dim * history_len
    tail_dim = max(0, total_dim - core_dim)

    print(f"[STEP {step_count}] obs_env{env_id} total_dim={total_dim}, history_len={history_len}, per_step_dim={per_step_dim}, tail_dim={tail_dim}")

    for hist_idx in range(history_len):
        print(f"[STEP {step_count}] obs_env{env_id} history[{hist_idx}]")
        base = hist_idx * per_step_dim
        offset = 0
        for name, dim in per_step_fields:
            start = base + offset
            end = start + dim
            values = obs_env[start:end].tolist()
            print(f"[STEP {step_count}]   {name:<30} {values}")
            offset += dim

    if tail_dim > 0:
        tail = obs_env[core_dim:].tolist()
        print(f"[STEP {step_count}] obs_env{env_id} tail(height_scan_or_extra_features) {tail}")


def _print_joint_index_layout(env):
    joint_names = list(getattr(env.robot, "joint_names", []))
    if len(joint_names) == 0:
        print("[JOINT] No joint names found on env.robot.")
        return

    print(f"[JOINT] total joints = {len(joint_names)}")
    for idx, name in enumerate(joint_names):
        print(f"[JOINT] index={idx:>2} name={name}")


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    try:
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    except Exception as exc:
        print(f"[WARN] ONNX export failed, continue play without ONNX: {exc}")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    _print_policy_obs_layout(env, obs)
    _print_joint_index_layout(env)
    try:
        validate_symmetry_involution(env)
    except Exception as exc:
        print(f"[WARN] symmetry involution check failed/skipped: {exc}")
    print_flag = 0
    print_every = 20
    if print_flag == 1:
        print(f"[PLAY] Printing env0 action, applied_torque, and grouped obs every {print_every} steps.")
    step_count = 0

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if print_flag == 1 and step_count % max(1, print_every) == 0:
                action_env0 = actions[0].detach().cpu().tolist()
                torque_env0 = env.robot.data.applied_torque[0].detach().cpu().tolist()
                print(f"[STEP {step_count}] action_env0={action_env0}")
                print(f"[STEP {step_count}] torque_env0={torque_env0}")
                _print_obs_grouped(env, obs, step_count, env_id=0)
            step_count += 1


if __name__ == "__main__":
    play()
    simulation_app.close()
