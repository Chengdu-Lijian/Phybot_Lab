from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Phybot_miniCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        default_joint_angles = {  
            # left leg
            'left_hip_pitch': -0.345, # 30
            'left_hip_roll': -0.0, # 0
            'left_hip_yaw': -0.0, # 7
            'left_knee': 0.69, # 53
            'left_ankle_pitch': -0.341, # 27
            'left_ankle_roll': 0, # 0

            'right_hip_pitch': -0.345,
            'right_hip_roll': 0.00,
            'right_hip_yaw': 0.0,
            'right_knee': 0.69,
            'right_ankle_pitch': -0.341,
            'right_ankle_roll': 0,

            # waist
            # 'waist_yaw': 0.0,

            # 'left_shoulder_pitch': 0.0,
            # 'left_shoulder_roll': 0.0, # 10
            # 'left_shoulder_yaw': 0.0, # 10
            # 'left_elbow_pitch': 0,

            # 'right_shoulder_pitch': 0.0,
            # 'right_shoulder_roll': -0.0, #10
            # 'right_shoulder_yaw': -0.0, #10
            # 'right_elbow_pitch': 0,
        }
    
    class env(LeggedRobotCfg.env):
        num_actions = 12
        num_observations = 3+3+3+num_actions*3+2 # 47
        num_privileged_obs = 3+3+3+3+num_actions*3+2 # 50
        num_envs = 4096
        


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
            'hip_pitch': 150, 'hip_roll': 150, 'hip_yaw': 150,
            'knee': 150,
            'ankle_pitch': 150., 'ankle_roll': 50., 
            
            # 'waist_yaw': 150, 

            # 'shoulder_pitch': 150, 'shoulder_roll': 150, 'shoulder_yaw': 50,
            # 'elbow_pitch': 150
        }  
        damping = {
            'hip_pitch': 10, 'hip_roll': 10, 'hip_yaw': 10,
            'knee': 10,
            'ankle_pitch': 10., 'ankle_roll': 5., 

            # 'waist_yaw': 10, 

            # 'shoulder_pitch': 10, 'shoulder_roll': 10, 'shoulder_yaw': 5, 
            # 'elbow_pitch': 10
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/phybot_mini/urdf/phybot_mini_mark1_half.urdf'
        name = "mini_phybot"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["waist_yaw", 'knee']
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.65
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1
            ang_vel_xy = -0.05
            orientation = -2
            base_height = -20.0
            dof_acc = -3e-7
            dof_vel = -2e-3
            feet_air_time = 0.5
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -1.0
            alive = 0.3
            hip_pos = -0.7
            contact_no_vel = -0.5
            feet_swing_height = -20.0
            contact = 2.0
            torques = -0.0000005


            # adding 
            # feet orientation
            feet_orien_diff_osu = -5.0

class Phybot_miniCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'phybot_mini'

  
