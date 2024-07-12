import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.models.value_function_models.embd_value import EmbdValueNetwork
import gym
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space

class Agent:
    def __init__(self, args, cent_obs_space, act_spaces):
        self.model = EmbdValueNetwork(args, cent_obs_space=share_obs_space, act_spaces=act_space)

    def result(self, obs_all, act_all, agent_ids):
        return self.model(obs_all, act_all, agent_ids)
    
# Define the dimensions for share_obs_space and act_space
share_obs_dim = 54
act_dim = 5
num_agents = 3

# Create a Box space for share_obs_space and sample from it
share_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
share_obs_sample = share_obs_space.sample()

# Create a list of Box spaces for act_space and sample from each
act_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32) for _ in range(num_agents)]
act_samples = [space.sample() for space in act_space]

# Print the samples and their types
print(f"share_obs_sample: {share_obs_sample}, type: {type(share_obs_sample)}")
print(f"act_samples: {act_samples}, types: {[type(sample) for sample in act_samples]}")

new_args = dict()
new_args["num_policies"] = 8
new_args["num_heads"] = 4
new_args["num_agents"] = num_agents
new_args["embedding_dim"] = 9
new_args["output_dim"] = 1

# Example inputs
obs_all = torch.randn(2, share_obs_dim)  # (batch_size, num_agents * obs_dim)
act_all = torch.randn(2, act_dim * num_agents)  # (batch_size, num_agents * action_dim)
agent_ids = torch.tensor([0, 1, 2])  # agent IDs

# Create Agent objects for each agent
agents = []
agents.append(Agent(new_args, cent_obs_space=share_obs_space, act_spaces=act_space))
new_args['model'] = agents[0].model
for i in range(1, new_args['num_agents']):
    agents.append(Agent(new_args, cent_obs_space=share_obs_space, act_spaces=act_space))

# Assuming all agents use the same model, perform forward pass
for i, agent in enumerate(agents):
    result = agent.result(obs_all, act_all, agent_ids)
    print(f'i:{i} result.size():{result.size()}')  # Expected size: (batch_size, action_dim)
    print(result)  # Expected size: (batch_size, action_dim)