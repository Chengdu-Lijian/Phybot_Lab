import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
from pynput import keyboard
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    frequency = 0.8  # Initial frequency
    foot_height = 0.1  # Initial foot height
    root_height = 0.0  # Initial root height
    root_pitch = 0.0  # Initial root pitch angle
    waist_angle = 0.0  # Initial waist angle
    mode = 0  # Initial mode (0-4)

def on_press(key):
    speed_increment = 0.1
    try:
        # Adjust speed
        if key == keyboard.Key.up:  # Increase vx
            cmd.vx += speed_increment
        elif key == keyboard.Key.down:  # Decrease vx
            cmd.vx -= speed_increment
        elif key == keyboard.Key.left:  # Increase vy
            cmd.vy += speed_increment
        elif key == keyboard.Key.right:  # Decrease vy
            cmd.vy -= speed_increment
        if key.char == ',':  # Increase dyaw
            cmd.dyaw += speed_increment
        elif key.char == '.':  # Decrease dyaw
            cmd.dyaw -= speed_increment
        
        # Adjust other parameters
        elif key.char == 'f':  # Increase frequency
            cmd.frequency += 0.1
        elif key.char == 'd':  # Decrease frequency
            cmd.frequency -= 0.1
        elif key.char == 'h':  # Increase foot height
            cmd.foot_height += 0.01
        elif key.char == 'g':  # Decrease foot height
            cmd.foot_height -= 0.01
        elif key.char == 'r':  # Increase root height
            cmd.root_height += 0.01
        elif key.char == 't':  # Decrease root height
            cmd.root_height -= 0.01
        elif key.char == 'p':  # Increase root pitch angle
            cmd.root_pitch += 0.087
        elif key.char == 'o':  # Decrease root pitch angle
            cmd.root_pitch -= 0.087
        elif key.char == 'w':  # Increase waist angle
            cmd.waist_angle += 0.087
        elif key.char == 'e':  # Decrease waist angle
            cmd.waist_angle -= 0.087
        elif key.char == 'm':  # Change mode (cyclic 0-4)
            cmd.mode = (cmd.mode + 1) % 4

        print(f"Current frequency: {cmd.frequency}")
        print(f"Current foot height: {cmd.foot_height}")
        print(f"Current root height: {cmd.root_height}")
        print(f"Current root pitch: {cmd.root_pitch}")
        print(f"Current waist angle: {cmd.waist_angle}")
        print(f"Current mode: {cmd.mode}")

    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # 停止监听
        return False


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]  # min max [m/s]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]  # min max [m/s]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        env.commands[:, 0] = cmd.vx
        env.commands[:, 1] = cmd.vy
        env.commands[:, 2] = cmd.dyaw
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
