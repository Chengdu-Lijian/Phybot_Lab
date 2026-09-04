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

from legged_lab.envs.phybot_c2.phybot_c2_env import PhybotC2
from legged_lab.envs.phybot_c2.walk_cfg import (
    PhybotC2WalkAgentCfg,
    PhybotC2WalkFlatEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("phybot_walk", PhybotC2, PhybotC2WalkFlatEnvCfg(), PhybotC2WalkAgentCfg())
