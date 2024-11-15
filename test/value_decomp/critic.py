# critic.py

import torch
import torch.nn as nn
from config import device

class Critic(nn.Module):
    def __init__(self, obs_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.to(device)
        else:
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
        value = self.critic(obs)
        return value

class ValueDecompositionCritic(nn.Module):
