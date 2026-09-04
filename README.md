# Phybot Lab

## Overview

This repository contains reinforcement learning-based locomotion environments for the PHYBOT humanoid robot (c2), built on Isaac Lab v2.1.0 and Isaac Sim v4.5.0.

---
This repository requires:

- `Isaac Lab 2.1`
- `Isaac Sim 4.5`

The commands below are intended to be run from the repository root.

## Installation

### 1. Install Isaac Lab

Install Isaac Lab by following the official guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Important:
- Isaac Lab version: v2.1.0
- Isaac Sim version: v4.5.0
- Conda installation is recommended to simplify running Python scripts from the terminal.

---

### 2. Clone This Repository

Clone or copy this repository outside the IsaacLab directory:



---

### 3. Install the PHYBOT Library (Editable Mode)

Using a python interpreter that has Isaac Lab installed, install the library

    cd Phybot-Lab
    pip install -e .

---

### 4. Install RSL-RL 

    cd Phybot-Lab/rsl_rl
    pip install -e .

---

## Task

Only one task is currently registered:

- `phybot_walk`

Use `--task=phybot_walk` for training, playback, AMP motion playback, and sim-to-sim execution.

## Training

Train the Phybot policy:

```bash
python legged_lab/scripts/train.py --task=phybot_walk --headless --logger=tensorboard --num_envs=4096
```

Common arguments:

- `--task`: registered task name
- `--headless`: disable GUI rendering
- `--num_envs`: number of parallel environments
- `--seed`: random seed
- `--logger`: logging backend, such as `tensorboard`, `wandb`, or `neptune`
- `--max_iterations`: override the configured training iterations
- `--experiment_name`: output directory name under `logs/`
- `--run_name`: suffix for the current run
- `--resume`: resume from a checkpoint
- `--load_run`: choose a previous run directory
- `--checkpoint`: choose a specific checkpoint file
- `--log_project_name`: project name for `wandb` or `neptune`
- `--distributed`: enable distributed training

## Policy Playback

Replay a trained policy:

```bash
python legged_lab/scripts/play.py --task=phybot_walk --num_envs=1
```

Common arguments:

- `--task`: registered task name
- `--num_envs`: number of playback environments
- `--seed`: random seed
- `--headless`: disable GUI playback
- `--load_run` and `--checkpoint`: load a specific trained policy

## AMP Motion Playback And Export

Play the default visualization motion configured for Phybot:

```bash
python legged_lab/scripts/play_amp_animation.py --task=phybot_walk --num_envs=1
```

Play a custom motion file from `legged_lab/envs/phybot_c2/datasets/motion_visualization/`:

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=phybot_walk \
  --num_envs=1 \
  --phybot_motion run_15_50hz
```

You can also pass an absolute path:

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=phybot_walk \
  --num_envs=1 \
  --phybot_motion /abs/path/to/run_15_50hz.json
```

Play only a frame range:

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=phybot_walk \
  --num_envs=1 \
  --phybot_motion run_15_50hz \
  --frame_start 100 \
  --frame_end 260
```

Loop playback:

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=phybot_walk \
  --num_envs=1 \
  --phybot_motion run_15_50hz \
  --loop_play
```

Export AMP expert data:

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=phybot_walk \
  --num_envs=1 \
  --phybot_motion run_15_50hz \
  --save_path legged_lab/envs/phybot_c2/datasets/motion_amp_expert/motion.txt
```

Common arguments:

- `--task`: registered task name
- `--num_envs`: number of environments, usually `1` for playback
- `--fps`: target playback or export frame rate
- `--save_path`: output path for exported motion data
- `--loop_play`: loop until the viewer is closed
- `--phybot_motion`: motion file name or full path
- `--frame_start`: inclusive start frame
- `--frame_end`: exclusive end frame
- `--input_motion_path`: deprecated alias of `--phybot_motion`

Notes:

- `--loop_play` cannot be used together with `--save_path`.
- If `--phybot_motion` is given as a file name, the script looks under `legged_lab/envs/phybot_c2/datasets/motion_visualization/`.
- If the `.json` suffix is omitted, it is added automatically.
- Custom `phybot_motion` files are parsed as 51-dimensional frames:
  `base_lin_vel(3) + base_ang_vel(3) + gravity(3) + joint_pos(21) + joint_vel(21)`.


## Sim-to-Sim

Run MuJoCo sim-to-sim with the Phybot preset:

```bash
python legged_lab/scripts/sim2sim.py --task phybot_walk --duration 100
```

Common arguments:

- `--task`: fixed to `phybot_walk`
- `--policy`: optional exported policy file path or `exported/` directory; if omitted or invalid, sim2sim auto-selects the latest `logs/phybot_walk/*/exported/policy.pt`
- `--model`: optional MuJoCo XML path
- `--duration`: simulation duration in seconds
