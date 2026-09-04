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

import math
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.phybot_c2 import PHYBOT_C2_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    enable: bool = False
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class LiteRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"std": 0.8})
    # track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    track_lin_vel_xy_l2 = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_l2, weight=-2.0)
    # track_ang_vel_z_l2 = RewTerm(func=mdp.track_ang_vel_z_world_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    energy = RewTerm(func=mdp.energy, weight=-5e-4)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
            "contact_sensor", body_names=[
                    "left_knee", "right_knee", 
                    ".*_shoulder_.*", 
                    ".*_elbow_pitch", 
                    "base_link", "waist_yaw"
                ]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")}, weight=-1.2
    )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.6)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
    #     },
    # )
    # feet_force = RewTerm(
    #     func=mdp.body_force,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll"),
    #         "threshold": 500,
    #         "max_reward": 400,
    #     },
    # )
    # feet_stumble = RewTerm(
    #     func=mdp.feet_stumble,
    #     weight=0.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll"])},
    # )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)


def _collect_phybot_amp_motion_files() -> list[str]:
    dataset_root = Path(__file__).resolve().parent / "datasets" / "motion_amp_expert"
    search_dirs = [dataset_root / "new_traj_walk", dataset_root]
    for dataset_dir in search_dirs:
        json_files = sorted(dataset_dir.glob("*.json"))
        if json_files:
            return [str(file) for file in json_files]

        txt_files = sorted(dataset_dir.glob("*.txt"))
        if txt_files:
            return [str(file) for file in txt_files]

    return [str(dataset_root / "walk.txt")]


def _collect_phybot_display_motion_files() -> list[str]:
    dataset_dir = Path(__file__).resolve().parent / "datasets" / "motion_visualization"
    preferred_files = ["walk_06_50hz.json", "walk_07_50hz.json", "walk_back_10_50hz.json"]
    preferred_paths = [dataset_dir / name for name in preferred_files]
    if all(path.exists() for path in preferred_paths):
        return [str(path) for path in preferred_paths]

    json_files = sorted(dataset_dir.glob("walk*.json"))
    if json_files:
        return [str(file) for file in json_files]

    txt_files = sorted(dataset_dir.glob("walk*.txt"))
    if txt_files:
        return [str(file) for file in txt_files]

    any_json = sorted(dataset_dir.glob("*.json"))
    if any_json:
        return [str(file) for file in any_json]

    any_txt = sorted(dataset_dir.glob("*.txt"))
    if any_txt:
        return [str(file) for file in any_txt]

    return [str(dataset_dir / "walk.txt")]


@configclass
class PhybotC2WalkFlatEnvCfg:
    amp_motion_files_display = _collect_phybot_display_motion_files()
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=PHYBOT_C2_CFG,
        # terrain_type="generator",
        # terrain_generator=GRAVEL_TERRAINS_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="base_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.2,
        terminate_contacts_body_names=[
            "left_knee", "right_knee",     
            ".*_shoulder_.*",              
            ".*_elbow_pitch",              
            "base_link",
            "waist_yaw"
        ],
        feet_body_names=[".*_ankle_roll"], 
    ) 

    reward = LiteRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=2.0,
            ang_vel=0.25,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,
            actions=0.15,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.3,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.3, 0.9), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.15,
            projected_gravity=0.03,
            joint_pos=0.1,
            joint_vel=0.05,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)},
                    "velocity_range": {
                        "x": (-0.2, 0.2),
                        "y": (-0.2, 0.2),
                        "z": (-0.1, 0.1),
                        "roll": (-0.2, 0.2),
                        "pitch": (-0.2, 0.2),
                        "yaw": (-0.2, 0.2),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.8, 1.2),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 10.0),
                params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class PhybotC2WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", 
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "data_augmentation_func": "legged_lab.envs.phybot_c2.symmetry:get_symmetric_states",
            "mirror_loss_coeff": 0.0,
        },
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "phybot_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "phybot_walk"
    wandb_project = "phybot_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.9
    amp_motion_files = _collect_phybot_amp_motion_files()
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.8
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 21
