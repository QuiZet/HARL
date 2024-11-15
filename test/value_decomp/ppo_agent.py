# ppo_agent.py

import torch
import torch.nn as nn
from config import device

class PPOAgent(nn.Module):
    def __init__(self, obs_size, act_size):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, act_size),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        raise NotImplementedError

    def get_action(self, obs):
        try:
            if isinstance(obs, torch.Tensor):
                obs = obs.to(device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)  # Add batch dimension
            probs = self.actor(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            # Extract scalar values
            return action.cpu().numpy()[0], dist.log_prob(action)[0], dist.entropy()[0]
        except Exception as e:
            print(f"Error in get_action: {e}")
            return None, None, None

    def get_value(self, obs):
        try:
            if isinstance(obs, torch.Tensor):
                obs = obs.to(device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)  # Add batch dimension
            value = self.critic(obs)
            return value
        except Exception as e:
            print(f"Error in get_value: {e}")
            return None
