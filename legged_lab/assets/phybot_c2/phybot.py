import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

PHYBOT_C2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/phybot_c2/usd/phybot_c2_s_34.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
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
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw", ".*_knee"
            ],
            effort_limit_sim={
                ".*_hip_pitch": 200.0,
                ".*_hip_roll": 80.0,
                ".*_hip_yaw": 58.0,
                ".*_knee": 80.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch": 99.21,
                ".*_hip_roll": 98.90,
                ".*_hip_yaw": 98.37,
                ".*_knee": 98.90,
            },
            stiffness={
                ".*_hip_pitch": 100.0,
                ".*_hip_roll": 100.0,
                ".*_hip_yaw": 100.0,
                ".*_knee": 100.0,
            },
            damping={
                ".*_hip_pitch": 10.0,
                ".*_hip_roll": 10.0,
                ".*_hip_yaw": 10.0,
                ".*_knee": 10.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch", ".*_ankle_roll"],
            effort_limit_sim=58.0,
            velocity_limit_sim=98.37,
            stiffness={
                ".*_ankle_pitch": 50.0,
                ".*_ankle_roll": 50.0,
            },
            damping={
                ".*_ankle_pitch": 5.0,
                ".*_ankle_roll": 5.0,
            },
        ),
        "torso_and_arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw",
                ".*_shoulder_pitch",
                ".*_shoulder_roll",
                ".*_shoulder_yaw",
                ".*_elbow_pitch"
            ],
            effort_limit_sim={
                "waist_yaw": 58.0,
                ".*_shoulder_pitch": 58.0,
                ".*_shoulder_roll": 58.0,
                ".*_shoulder_yaw": 58.0,
                "left_elbow_pitch": 9.0,
                "right_elbow_pitch": 58.0,
            },
            velocity_limit_sim={
                "waist_yaw": 98.37,
                ".*_shoulder_pitch": 98.37,
                ".*_shoulder_roll": 98.37,
                ".*_shoulder_yaw": 98.37,
                "left_elbow_pitch": 98.16,
                "right_elbow_pitch": 98.37,
            },            
            
            stiffness={
                "waist_yaw": 100.0,
                ".*_shoulder_pitch": 50.0,
                ".*_shoulder_roll": 50.0,
                ".*_shoulder_yaw": 5.0,
                ".*_elbow_pitch": 50.0,
            },
            damping={
                "waist_yaw": 10.0,
                ".*_shoulder_pitch": 5.0,
                ".*_shoulder_roll": 5.0,
                ".*_shoulder_yaw": 5.0,
                ".*_elbow_pitch": 5.0,
            },
        ),
    },
)
