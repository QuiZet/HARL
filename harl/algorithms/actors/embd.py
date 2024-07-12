"""Transferable Multi-Agent Reinforcement Learning with Dynamic Participating Agents
 algorithm."""
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.algorithms.actors.hatd3 import HATD3
from harl.models.policy_models.embd_policy import EmbdPolicyNetwork
from harl.utils.envs_tools import get_shape_from_obs_space

import inspect

class EMBD(HATD3):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        print('EMBD')
        print(f'args:{args}')
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        act_dim = act_space.shape[0]
        print(f'hidden_sizes:{hidden_sizes}')
        print(f'activation_func:{activation_func}')
        print(f'final_activation_func:{final_activation_func}')
        print(f'obs_shape:{obs_shape} {obs_shape[0]}')
        print(f'act_dim:{act_dim}')

        # Modify the actor policy
        self.actor = EmbdPolicyNetwork(input_dim=obs_shape[0], 
                                       num_heads=4,
                                       num_policies=8,
                                       embedding_dim=14,
                                       output_dim=act_dim, 
                                       num_agents=args["num_agents"], 
                                       device=device)

        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def get_target_actions(self, obs, obs_all, agent_ids):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        obs_all = check(obs_all).to(**self.tpdv)
        agent_ids = check(agent_ids).to(obs_all.device)
        #print(f'obs:{obs.shape} obs_all:{obs_all.shape} agent_ids:{agent_ids.shape}')
        #agent_ids = torch.arange(len(obs)).to(obs.device)

        actions = self.target_actor(obs_all, agent_ids)
        noise = torch.randn_like(actions) * self.policy_noise * self.scale
        noise = torch.clamp(
            noise, -self.noise_clip * self.scale, self.noise_clip * self.scale
        )
        actions += noise
        actions = torch.clamp(actions, self.low, self.high)
        #print(f'actions:{actions.shape} obs:{obs.shape}')
        return actions

    def get_actions(self, obs, add_noise, obs_all, agent_ids):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        #print(f'EMBD::get_actions :{inspect.stack()[1].function}')
        #print(f'obs_all:{len(obs_all)} agent_ids {agent_ids}')

        obs = check(obs).to(**self.tpdv)
        obs_all = check(obs_all).to(**self.tpdv)
        # check that the tensor is in the form (batch, num_agents, dim)
        obs_all = obs_all.unsqueeze(0) if obs_all.dim() == 2 else obs_all
        agent_ids = check(agent_ids).to(obs_all.device)
        #print(f'obs:{obs.shape} obs_all:{obs_all.shape} agent_ids:{agent_ids.shape}')
        #agent_ids = torch.arange(len(obs)).to(obs.device)
        actions = self.actor(obs_all, agent_ids)
        #print(f'actions s:{actions.shape}')
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        #print(f'actions:{actions}')
        return actions