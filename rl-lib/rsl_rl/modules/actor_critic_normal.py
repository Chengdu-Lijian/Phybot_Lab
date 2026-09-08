import numpy as np 

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

import torch.cuda.amp as amp


class ActorCritic_Normal(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs, # 79 +210
                        num_critic_obs, # 79 + 210 + 111
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_std=1,
                        **kwargs):
        # if kwargs:
        #     print("ActorCritic_Normal.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_Normal, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs  
        mlp_input_dim_c = num_critic_obs

        class Actor(nn.Module):
            def __init__(self, mlp_input_dim_a, num_actions,
                         actor_hidden_dims, activation):
                super().__init__()

                if len(actor_hidden_dims) > 0:
                    actor_layers = []
                    actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
                    actor_layers.append(activation)
                    for l in range(len(actor_hidden_dims)):
                        if l == len(actor_hidden_dims) - 1:
                            actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                            actor_layers.append(nn.Hardtanh(min_val=-5.0, max_val=5.0))
                        else:
                            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                            actor_layers.append(activation)
                    self.actor_backbone = nn.Sequential(*actor_layers)
                else:
                    self.actor_backbone = nn.Identity()

            def forward(self, obs):

                obs_prop = obs[:, :mlp_input_dim_a] 
                backbone_input = torch.cat([obs_prop], dim=1) 
                backbone_output = self.actor_backbone(backbone_input)  
                return backbone_output

            
        self.actor = Actor(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)

        class Critic(nn.Module):
            def __init__(self, mlp_input_dim_c, critic_hidden_dims, 
                         activation):
                super().__init__()

                # Value
                if len(critic_hidden_dims) > 0:
                    critic_layers = []
                    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                    critic_layers.append(activation)
                    for l in range(len(critic_hidden_dims)):
                        if l == len(critic_hidden_dims) - 1:
                            critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                        else:
                            critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                            critic_layers.append(activation)
                    self.critic_backbone = nn.Sequential(*critic_layers)
                else:
                    self.critic_backbone = nn.Identity()

            def forward(self, obs):
                with torch.autocast(device_type="cuda", dtype=torch.float16):  
                    prop_and_priv = obs[:, :mlp_input_dim_c]
                    backbone_output = self.critic_backbone(prop_and_priv)
                    return backbone_output
                return backbone_output


        self.critic = Critic(mlp_input_dim_c, \
                             critic_hidden_dims, activation)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(torch.tensor(init_std))
        self.distribution = None
        Normal.set_default_validate_args = False


    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations,**kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def actor_inference(self, observations):
        actor_output = self.actor.actor_backbone(observations)
        return actor_output
    
    def history_encode(self, obs_history):
        latent = self.actor.history_encoder(obs_history.detach())
        return latent

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
