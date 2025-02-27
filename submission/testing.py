###########################################################################################
#      This code is a part of the submission for the CMP9140 Reserch Project @ UoL,2024   #
#   Aim : Embodied Intelligence: The embodiment of Bio-Inspired Physical Intelligence     #  
# Objective : reinforcemnet Learnig based traning for differnt morphologies utilizing     #
# different mechanical tail to achieve stable gait motion                                 #  
# The assisment was done under the supervision of Dr. Alexander Klimchik, SoCS, UoL       #
#For indepth understanding of usability of the research and the code refer to study matter# 
###########################################################################################
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from stable_baselines3.common.callbacks import EvalCallback

#Below is the path for the modified XML model
path_to_xml= ''


#Define paths for logs and model saves
log_dir = "./tb_ant_logs/"
model_save_path = "ppo_ant_real"

#Create directories if they do not exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)


seed = random.randint(0,1000) 
learning_rate = 0.0003

#Custom policy to introduce policy network and value network 
#This is an attempt to pelicate the policiy networl from (Henderson ,2019)
#Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D., 2019. Deep Reinforcement Learning that Matters. 
#The code is available from : https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


#Create environment
env = gym.make('Hopper-v4',xml_file = path_to_xml)# ctrl_cost_weight=0.1= ,xml_file = , reset_noise_scale = ....)

#Create model with TensorBoard callback
model = PPO(CustomActorCriticPolicy, env, verbose=1, device = 'cuda', tensorboard_log=log_dir)

#below is callback function the check the model reward against a constrain set by teacher
#Upon a match it shall register the specific state 
## origanally avaialable from from https://stable-baselines3.readthedocs.io/en/master/guide/callback.html
#instance for callback

callback = EvalCallback(env, best_model_save_path="./tb_logs/",
                             log_path="./tb_logs/", eval_freq=500,
                             deterministic=True, render=False)

#callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

#Train the model
model.learn(total_timesteps=500000, progress_bar=True, callback= callback)

# Save the model
model.save(model_save_path)

# Optionally: Load and evaluate
# model = PPO.load(model_save_path)
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
