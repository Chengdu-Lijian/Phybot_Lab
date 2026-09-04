from __future__ import annotations

from typing import Optional

import torch

JOINT_MIRROR_PAIRS = (
    ("left_hip_pitch", "right_hip_pitch"),
    ("left_hip_roll", "right_hip_roll"),
    ("left_hip_yaw", "right_hip_yaw"),
    ("left_knee", "right_knee"),
    ("left_ankle_pitch", "right_ankle_pitch"),
    ("left_ankle_roll", "right_ankle_roll"),
    ("left_shoulder_pitch", "right_shoulder_pitch"),
    ("left_shoulder_roll", "right_shoulder_roll"),
    ("left_shoulder_yaw", "right_shoulder_yaw"),
    ("left_elbow_pitch", "right_elbow_pitch"),
)

# The sign convention below follows URDF axis/origin symmetry under reflection
# over the sagittal plane (y -> -y): roll/yaw-like dofs negate, pitch-like dofs keep sign.
NEGATED_JOINTS = {
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_ankle_roll",
    "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
}


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _to_long_tensor(index_like, device: torch.device) -> torch.Tensor:
    if isinstance(index_like, torch.Tensor):
        return index_like.to(device=device, dtype=torch.long)
    return torch.as_tensor(index_like, device=device, dtype=torch.long)


def _build_joint_mirror_maps(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    num_actions = env.num_actions
    mirror_index = torch.arange(num_actions, device=device, dtype=torch.long)
    mirror_sign = torch.ones(num_actions, device=device)

    joint_names = list(getattr(env.robot.data, "joint_names", []))[:num_actions]
    name_to_id = {name: idx for idx, name in enumerate(joint_names)}

    swapped_pairs = 0
    for left_name, right_name in JOINT_MIRROR_PAIRS:
        if left_name in name_to_id and right_name in name_to_id:
            left_id = name_to_id[left_name]
            right_id = name_to_id[right_name]
            mirror_index[left_id] = right_id
            mirror_index[right_id] = left_id
            swapped_pairs += 1

    if swapped_pairs == 0:
        raise ValueError("No mirrored joint pairs were resolved for symmetry augmentation.")

    for joint_name in NEGATED_JOINTS:
        if joint_name in name_to_id:
            mirror_sign[name_to_id[joint_name]] = -1.0

    return mirror_index, mirror_sign


def _build_feet_swap_index(env, device: torch.device) -> torch.Tensor:
    feet_ids = _to_long_tensor(env.feet_cfg.body_ids, device)
    feet_swap_index = torch.arange(len(feet_ids), device=device, dtype=torch.long)

    if len(feet_ids) < 2:
        return feet_swap_index

    left_right_body_ids = _to_long_tensor(getattr(env, "feet_body_ids", feet_ids[:2]), device)
    if len(left_right_body_ids) < 2:
        return feet_swap_index

    left_body_id = int(left_right_body_ids[0].item())
    right_body_id = int(left_right_body_ids[1].item())
    feet_id_list = [int(x.item()) for x in feet_ids]

    if left_body_id in feet_id_list and right_body_id in feet_id_list:
        left_pos = feet_id_list.index(left_body_id)
        right_pos = feet_id_list.index(right_body_id)
        feet_swap_index[left_pos] = right_pos
        feet_swap_index[right_pos] = left_pos

    return feet_swap_index


def _get_cached_maps(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_mirror_joint_index") or not hasattr(env, "_mirror_joint_sign"):
        joint_index, joint_sign = _build_joint_mirror_maps(env, device)
        env._mirror_joint_index = joint_index
        env._mirror_joint_sign = joint_sign

    if not hasattr(env, "_mirror_feet_swap_index"):
        env._mirror_feet_swap_index = _build_feet_swap_index(env, device)

    joint_index = env._mirror_joint_index.to(device=device, dtype=torch.long)
    joint_sign = env._mirror_joint_sign.to(device=device, dtype=torch.float)
    feet_swap_index = env._mirror_feet_swap_index.to(device=device, dtype=torch.long)
    return joint_index, joint_sign, feet_swap_index


def _mirror_policy_frame(obs_frame: torch.Tensor, env) -> torch.Tensor:
    joint_index, joint_sign, _ = _get_cached_maps(env, obs_frame.device)
    num_actions = env.num_actions
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))

    out = obs_frame.clone()

    # root_ang_vel_b: [wx, wy, wz] -> [-wx, wy, -wz]
    out[..., 0] = -obs_frame[..., 0]
    out[..., 1] = obs_frame[..., 1]
    out[..., 2] = -obs_frame[..., 2]

    # projected_gravity_b: [gx, gy, gz] -> [gx, -gy, gz]
    out[..., 3] = obs_frame[..., 3]
    out[..., 4] = -obs_frame[..., 4]
    out[..., 5] = obs_frame[..., 5]

    # command: [lin_x, lin_y, yaw] -> [lin_x, -lin_y, -yaw]
    out[..., 6] = obs_frame[..., 6]
    out[..., 7] = -obs_frame[..., 7]
    out[..., 8] = -obs_frame[..., 8]

    joint_pos_start = 9
    joint_vel_start = joint_pos_start + num_actions
    action_start = joint_vel_start + num_actions

    out[..., joint_pos_start:joint_vel_start] = obs_frame[..., joint_pos_start:joint_vel_start][..., joint_index] * joint_sign
    out[..., joint_vel_start:action_start] = obs_frame[..., joint_vel_start:action_start][..., joint_index] * joint_sign
    out[..., action_start : action_start + num_actions] = (
        obs_frame[..., action_start : action_start + num_actions][..., joint_index] * joint_sign
    )

    if use_gait_phase:
        gait_start = action_start + num_actions
        # [sin_l, sin_r, cos_l, cos_r, ratio_l, ratio_r]
        out[..., gait_start : gait_start + 2] = obs_frame[..., gait_start : gait_start + 2][..., [1, 0]]
        out[..., gait_start + 2 : gait_start + 4] = obs_frame[..., gait_start + 2 : gait_start + 4][..., [1, 0]]
        out[..., gait_start + 4 : gait_start + 6] = obs_frame[..., gait_start + 4 : gait_start + 6][..., [1, 0]]

    return out


def _mirror_critic_frame(obs_frame: torch.Tensor, env) -> torch.Tensor:
    num_actions = env.num_actions
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))
    feet_dim = len(env.feet_cfg.body_ids)
    _, _, feet_swap_index = _get_cached_maps(env, obs_frame.device)

    actor_step_dim = 9 + 3 * num_actions + (6 if use_gait_phase else 0)
    critic_step_dim = actor_step_dim + 3 + feet_dim
    if obs_frame.shape[-1] != critic_step_dim:
        raise ValueError(f"Unexpected critic-step obs dim: {obs_frame.shape[-1]} != {critic_step_dim}")

    out = obs_frame.clone()
    out[..., :actor_step_dim] = _mirror_policy_frame(obs_frame[..., :actor_step_dim], env)

    # root_lin_vel_b: [vx, vy, vz] -> [vx, -vy, vz]
    lin_vel_start = actor_step_dim
    out[..., lin_vel_start] = obs_frame[..., lin_vel_start]
    out[..., lin_vel_start + 1] = -obs_frame[..., lin_vel_start + 1]
    out[..., lin_vel_start + 2] = obs_frame[..., lin_vel_start + 2]

    feet_start = actor_step_dim + 3
    out[..., feet_start : feet_start + feet_dim] = obs_frame[..., feet_start : feet_start + feet_dim][
        ..., feet_swap_index
    ]

    return out


def _augment_obs(obs: torch.Tensor, env, obs_type: str) -> torch.Tensor:
    num_actions = env.num_actions
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))
    actor_step_dim = 9 + 3 * num_actions + (6 if use_gait_phase else 0)

    if obs_type == "policy":
        history_len = env.cfg.robot.actor_obs_history_length
        step_dim = actor_step_dim
        mirror_frame_fn = _mirror_policy_frame
    elif obs_type == "critic":
        history_len = env.cfg.robot.critic_obs_history_length
        step_dim = actor_step_dim + 3 + len(env.feet_cfg.body_ids)
        mirror_frame_fn = _mirror_critic_frame
    else:
        raise ValueError(f"Unsupported obs_type={obs_type!r}. Expected 'policy' or 'critic'.")

    core_dim = history_len * step_dim
    if obs.shape[-1] < core_dim:
        raise ValueError(f"{obs_type} obs dim {obs.shape[-1]} is smaller than expected core dim {core_dim}.")

    obs_core = obs[..., :core_dim]
    obs_tail = obs[..., core_dim:]
    obs_core = obs_core.view(obs.shape[0], history_len, step_dim)
    obs_core_mirror = mirror_frame_fn(obs_core, env)
    obs_mirror = torch.cat((obs_core_mirror.reshape(obs.shape[0], core_dim), obs_tail), dim=-1)
    return torch.cat((obs, obs_mirror), dim=0)


def _augment_actions(actions: torch.Tensor, env) -> torch.Tensor:
    joint_index, joint_sign, _ = _get_cached_maps(env, actions.device)
    actions_mirror = actions[..., joint_index] * joint_sign
    return torch.cat((actions, actions_mirror), dim=0)


@torch.no_grad()
def get_symmetric_states(
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    env=None,
    obs_type: str = "policy",
):
    env = _unwrap_env(env)
    out_obs = None if obs is None else _augment_obs(obs, env, obs_type=obs_type)
    out_actions = None if actions is None else _augment_actions(actions, env)
    return out_obs, out_actions


@torch.no_grad()
def validate_symmetry_involution(
    env,
    batch_size: int = 64,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Quick sanity check: applying mirror twice should recover the original tensor."""
    env = _unwrap_env(env)
    device = env.device
    num_actions = int(env.num_actions)
    history_len = int(env.cfg.robot.actor_obs_history_length)
    use_gait_phase = bool(getattr(env, "use_gait_phase", False))

    step_dim = 9 + 3 * num_actions + (6 if use_gait_phase else 0)
    obs_dim = history_len * step_dim
    obs = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randn(batch_size, num_actions, device=device)

    obs_once, _ = get_symmetric_states(obs=obs, actions=None, env=env, obs_type="policy")
    obs_twice, _ = get_symmetric_states(obs=obs_once[batch_size:], actions=None, env=env, obs_type="policy")
    obs_recovered = obs_twice[batch_size:]

    _, actions_once = get_symmetric_states(obs=None, actions=actions, env=env, obs_type="policy")
    _, actions_twice = get_symmetric_states(obs=None, actions=actions_once[batch_size:], env=env, obs_type="policy")
    actions_recovered = actions_twice[batch_size:]

    obs_max_abs_err = (obs_recovered - obs).abs().max().item()
    actions_max_abs_err = (actions_recovered - actions).abs().max().item()
    obs_ok = torch.allclose(obs_recovered, obs, atol=atol, rtol=rtol)
    actions_ok = torch.allclose(actions_recovered, actions, atol=atol, rtol=rtol)
    passed = bool(obs_ok and actions_ok)

    print(
        "[SYMMETRY CHECK]",
        f"passed={passed}",
        f"obs_max_abs_err={obs_max_abs_err:.3e}",
        f"actions_max_abs_err={actions_max_abs_err:.3e}",
    )
    return {
        "passed": passed,
        "obs_ok": bool(obs_ok),
        "actions_ok": bool(actions_ok),
        "obs_max_abs_err": obs_max_abs_err,
        "actions_max_abs_err": actions_max_abs_err,
    }
