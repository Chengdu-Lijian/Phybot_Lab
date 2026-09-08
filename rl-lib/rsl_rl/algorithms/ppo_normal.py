# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic_Normal
from rsl_rl.storage import RolloutStorage_Normal

# from torch.amp import autocast, GradScaler

class PPO_NORMAL:
    actor_critic: ActorCritic_Normal
    def __init__(self,
                 actor_critic,

                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 
                 min_policy_std=None,
                 mirror_coef =1.0,
                 learning_rate_coef_schedual = [0, 10000, 1.e-5],
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.lr_coef_schedual = learning_rate_coef_schedual
        self.mirror_coef = mirror_coef#5.0
        print("self.mirror_coef: ", self.mirror_coef)

        # PPO_NORMAL components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.transition = RolloutStorage_Normal.Transition()

        # PPO_NORMAL parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.min_policy_std = torch.tensor(min_policy_std,device=self.device)

        self.counter = 0
        self.vel_loss_last = 1.0


        # self.scaler = GradScaler()


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage_Normal(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

            # Record the transition
            self.storage.add_transitions(self.transition)
        else:
            self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_mirror_loss = 0.0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch,hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # 希望policy面对镜像或相反处理的状态输入时，能得到相同的动作输出，使得左右半身运动更为对称
                if self.mirror_coef != 0:
                    new_action_batch = self.actor_critic.act_inference(obs_batch)
                    
                    #镜像处理obs
                    mirror_obs_batch = self._get_mirror_obs(obs_batch)
                    #根据镜像处理后的obs获得action
                    mirror_action_batch = self.actor_critic.act_inference(mirror_obs_batch)

                    mirror_action_batch_temp = mirror_action_batch.clone()
                    # Mirror leg actions
                    mirror_action_batch[:, 0:6] = mirror_action_batch_temp[:, 6:12]
                    mirror_action_batch[:, 6:12] = mirror_action_batch_temp[:, 0:6]
                    # get mirror action
                    # neg Leg Pitch; Leg Roll; Foot Roll
                    mirror_action_batch[:, 0] = - mirror_action_batch[:, 0]
                    mirror_action_batch[:, 6] = - mirror_action_batch[:, 6]
                    mirror_action_batch[:, 1] = - mirror_action_batch[:, 1]
                    mirror_action_batch[:, 7] = - mirror_action_batch[:, 7]
                    mirror_action_batch[:, 5] = - mirror_action_batch[:, 5]
                    mirror_action_batch[:, 11] = - mirror_action_batch[:, 11]
                    # neg Waist Yaw
                    mirror_action_batch[:, 12] = - mirror_action_batch[:, 12]
                    # Mirror arm actions
                    mirror_action_batch[:, 13:17] = mirror_action_batch_temp[:, 17:21]
                    mirror_action_batch[:, 17:21] = mirror_action_batch_temp[:, 13:17]
                    # neg Arm Roll; Arm Yaw
                    mirror_action_batch[:, 14] = - mirror_action_batch[:, 14]
                    mirror_action_batch[:, 15] = - mirror_action_batch[:, 15]
                    mirror_action_batch[:, 18] = - mirror_action_batch[:, 18]
                    mirror_action_batch[:, 19] = - mirror_action_batch[:, 19]

                    # mirror_loss = (mirror_action_batch - actions_batch).pow(2).mean()
                    mse_loss = nn.MSELoss()
                    mirror_loss = mse_loss(mirror_action_batch, new_action_batch)
                else:
                    mirror_loss = 0

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                elif self.schedule == 'linear':
                    self.learning_rate = max((1-self.counter/(self.lr_coef_schedual[1]-self.lr_coef_schedual[0]))*self.learning_rate_init,self.lr_coef_schedual[2])

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                if self.mirror_coef != 0:
                    loss = surrogate_loss \
                        + self.value_loss_coef * value_loss\
                        - self.entropy_coef * entropy_batch.mean()\
                        + self.mirror_coef * mirror_loss
                else:
                    loss = surrogate_loss \
                        + self.value_loss_coef * value_loss \
                        - self.entropy_coef * entropy_batch.mean()\

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()


                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if self.mirror_coef != 0:
                    mean_mirror_loss += mirror_loss.item()
                else:
                    mean_mirror_loss += mirror_loss                

        num_updates = self.num_learning_epochs * self.num_mini_batches
        print('num_updates: ', num_updates)
        mean_value_loss /= num_updates
        print('mean_value_loss: ', mean_value_loss)
        mean_surrogate_loss /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()

        self.update_counter()

        self.enforce_min_std()

        return mean_value_loss, mean_surrogate_loss, mean_mirror_loss
    
    def update_counter(self):
        self.counter += 1
    
    def enforce_min_std(self):
        current_std = self.actor_critic.std.detach()
        new_std = torch.max(current_std, self.min_policy_std).detach()
        self.actor_critic.std.data = new_std
        
        
    # 镜像或相反输入信息，使得机器人左右半身的运动更对称
    def _get_mirror_obs(self, obs_batch):
        mirror_obs_batch = obs_batch.clone()

        # mirror base_angle_vel
        # neg roll yaw
        mirror_obs_batch[:, 0] = -mirror_obs_batch[:, 0]
        mirror_obs_batch[:, 1] = mirror_obs_batch[:, 1]
        mirror_obs_batch[:, 2] = - mirror_obs_batch[:, 2]

        # mirror commands
        # neg y, yaw
        mirror_obs_batch[:, 3] = mirror_obs_batch[:, 3]
        mirror_obs_batch[:, 4] = - mirror_obs_batch[:, 4]
        mirror_obs_batch[:, 5] = - mirror_obs_batch[:, 5]
        
        # mirror sin & cos
        mirror_obs_batch[:, 6] = obs_batch[:, 7]
        mirror_obs_batch[:, 7] = obs_batch[:, 6]
        mirror_obs_batch[:, 8] = obs_batch[:, 8]
        mirror_obs_batch[:, 9] = obs_batch[:, 9]

        # mirror phase ratio
        mirror_obs_batch[:, 10] = obs_batch[:, 11]
        mirror_obs_batch[:, 11] = obs_batch[:, 10]

        # mirror leg_dof_pos
        mirror_obs_batch[:, 12:18] = obs_batch[:, 18:24]
        mirror_obs_batch[:, 18:24] = obs_batch[:, 12:18]

        # neg leg pitch, leg roll and foot yaw dof_pos
        mirror_obs_batch[:, 12] = - mirror_obs_batch[:, 12]
        mirror_obs_batch[:, 13] = - mirror_obs_batch[:, 13]
        mirror_obs_batch[:, 17] = - mirror_obs_batch[:, 17]
        mirror_obs_batch[:, 18] = - mirror_obs_batch[:, 18]
        mirror_obs_batch[:, 19] = - mirror_obs_batch[:, 19]
        mirror_obs_batch[:, 23] = - mirror_obs_batch[:, 23]

        # mirror waist roll
        mirror_obs_batch[:,24] = obs_batch[:, 24]
        # neg yaw roll
        mirror_obs_batch[:, 24] = - mirror_obs_batch[:, 24]

        # mirror arm_dof_pos
        mirror_obs_batch[:, 25:29] = obs_batch[:, 29:33]
        mirror_obs_batch[:, 29:33] = obs_batch[:, 25:29]

        # neg arm_dof_pos roll yaw
        mirror_obs_batch[:, 26] = - mirror_obs_batch[:, 26]
        mirror_obs_batch[:, 27] = - mirror_obs_batch[:, 27]
        mirror_obs_batch[:, 30] = - mirror_obs_batch[:, 30]
        mirror_obs_batch[:, 31] = - mirror_obs_batch[:, 31]

        # mirror dof_vel
        mirror_obs_batch[:, 33:39] = obs_batch[:, 39:45]
        mirror_obs_batch[:, 39:45] = obs_batch[:, 33:39]

        # neg leg pitch, leg roll and foot yaw dof_vel 
        mirror_obs_batch[:, 33] = - mirror_obs_batch[:, 33]
        mirror_obs_batch[:, 34] = - mirror_obs_batch[:, 34]
        mirror_obs_batch[:, 38] = - mirror_obs_batch[:, 38]

        mirror_obs_batch[:, 39] = - mirror_obs_batch[:, 39]
        mirror_obs_batch[:, 40] = - mirror_obs_batch[:, 40]
        mirror_obs_batch[:, 44] = - mirror_obs_batch[:, 44]

        # mirror waist
        mirror_obs_batch[:, 45] = obs_batch[:, 45]
        # neg yaw roll
        mirror_obs_batch[:, 45] = - mirror_obs_batch[:, 45]

        # mirror arm_dof_vel
        mirror_obs_batch[:, 46:50] = obs_batch[:, 50:54]
        mirror_obs_batch[:, 50:54] = obs_batch[:, 46:50]

        # neg arm_dof_vel roll yaw
        mirror_obs_batch[:, 47] = - mirror_obs_batch[:, 47]
        mirror_obs_batch[:, 48] = - mirror_obs_batch[:, 48]
        mirror_obs_batch[:, 51] = - mirror_obs_batch[:, 51]
        mirror_obs_batch[:, 52] = - mirror_obs_batch[:, 52]

        # mirror previous_actions_leg
        mirror_obs_batch[:, 54:60] = obs_batch[:, 60:66]
        mirror_obs_batch[:, 60:66] = obs_batch[:, 54:60]

        # neg actions_leg roll yaw
        mirror_obs_batch[:, 54] = - mirror_obs_batch[:, 54]
        mirror_obs_batch[:, 55] = - mirror_obs_batch[:, 55]
        mirror_obs_batch[:, 59] = - mirror_obs_batch[:, 59]
        mirror_obs_batch[:, 60] = - mirror_obs_batch[:, 60]
        mirror_obs_batch[:, 61] = - mirror_obs_batch[:, 61]
        mirror_obs_batch[:, 65] = - mirror_obs_batch[:, 65]

        # mirror waist
        mirror_obs_batch[:, 66] = obs_batch[:, 66]
        # neg yaw roll
        mirror_obs_batch[:, 66] = - mirror_obs_batch[:, 66]
        

        # mirror arm_action
        mirror_obs_batch[:, 67:71] = obs_batch[:, 71:75]
        mirror_obs_batch[:, 71:75] = obs_batch[:, 67:71]

        # neg arm_action roll yaw
        mirror_obs_batch[:, 68] = - mirror_obs_batch[:, 68]
        mirror_obs_batch[:, 69] = - mirror_obs_batch[:, 69]
        mirror_obs_batch[:, 72] = - mirror_obs_batch[:, 72]
        mirror_obs_batch[:, 73] = - mirror_obs_batch[:, 73]
        
        # mirror base_angle
        # neg roll and yaw
        mirror_obs_batch[:, 75] = -mirror_obs_batch[:, 75]
        mirror_obs_batch[:, 76] = mirror_obs_batch[:, 76]
        mirror_obs_batch[:, 77] = -mirror_obs_batch[:, 77]  

        # # mirror avg_vel
        # # neg y yaw vel
        # mirror_obs_batch[:, 78] = mirror_obs_batch[:, 78]
        # mirror_obs_batch[:, 79] = - mirror_obs_batch[:, 79]
        # mirror_obs_batch[:, 80] = - mirror_obs_batch[:, 80]              

        # start = 81
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]

        # start = 102
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]
        
        # start = 123
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]


        # start = 144
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]

        # start = 165
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]
       

        # start = 186
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]



        # start = 207
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]



        # start = 228
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]


        # start = 249
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]


        # start = 270
        # mirror_obs_batch[:, start:start+6] = obs_batch[:, start+6:start+12]
        # mirror_obs_batch[:, start+6:start+12] = obs_batch[:, start:start+6]

        # # neg leg_dof_pos roll yaw
        # mirror_obs_batch[:, start] = - mirror_obs_batch[:, start]
        # mirror_obs_batch[:, start+1] = - mirror_obs_batch[:, start+1]
        # mirror_obs_batch[:, start+5] = - mirror_obs_batch[:, start+5]
        # mirror_obs_batch[:, start+6] = - mirror_obs_batch[:, start+6]
        # mirror_obs_batch[:, start+7] = - mirror_obs_batch[:, start+7]
        # mirror_obs_batch[:, start+11] = - mirror_obs_batch[:, start+11]

        # # mirror waist yaw pitch roll
        # mirror_obs_batch[:,start+12] = obs_batch[:, start+12]
        # # neg yaw roll
        # mirror_obs_batch[:, start+12] = - mirror_obs_batch[:, start+12]

        # # mirror arm_dof_pos
        # mirror_obs_batch[:, start+13:start+17] = obs_batch[:, start+17:start+21]
        # mirror_obs_batch[:, start+17:start+21] = obs_batch[:, start+13:start+17]

        # # neg arm_dof_pos roll yaw
        # mirror_obs_batch[:, start+14] = - mirror_obs_batch[:, start+14]
        # mirror_obs_batch[:, start+15] = - mirror_obs_batch[:, start+15]
        # mirror_obs_batch[:, start+18] = - mirror_obs_batch[:, start+18]
        # mirror_obs_batch[:, start+19] = - mirror_obs_batch[:, start+19]

        return mirror_obs_batch