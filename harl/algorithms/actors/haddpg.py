"""HADDPG algorithm."""
from copy import deepcopy
import torch
from harl.models.policy_models.deterministic_policy import DeterministicPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase

import pandas as pd

class HADDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs)







class HADDPG(OffPolicyBase):
    file_counter = 0  # Class variable for generating unique file names

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

        # Generate a unique file name for saving data
        HADDPG.file_counter += 1
        self.file_name = f'haddpg_data_{HADDPG.file_counter}.csv'
        
        # Open the file in append mode
        self.file_handle = open(self.file_name, 'a')
        #self.save_data_header()  # Save header

    def get_actions(self, obs, add_noise):
        """Get actions for observations."""
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        #self.save_data(obs, actions, self.get_target_actions(obs))  # Save data
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations."""
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs)

    def save_data_header(self):
        """Save header to the CSV file."""
        header = 'obs,actions,target_actions\n'
        self.file_handle.write(header)

    def save_data(self, obs, actions, target_actions):
        """Save observations, actions, and target actions to the CSV file."""
        obs = obs.cpu().numpy()
        actions = actions.cpu().numpy()
        target_actions = target_actions.cpu().numpy()
        
        for o, a, t in zip(obs, actions, target_actions):
            line = f'{list(o)},{list(a)},{list(t)}\n'
            self.file_handle.write(line)
        self.file_handle.flush()

    def __del__(self):
        """Ensure the file is properly closed when the object is destroyed."""
        self.file_handle.close()
        print(f"Data saved to {self.file_name}")