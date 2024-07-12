import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.models.policy_models.embd_policy import EmbdPolicyNetwork

class Agent:
    def __init__(self, args):
        self.model = EmbdPolicyNetwork(args)

    def select_action(self, obs_all, agent_ids):
        return self.model(obs_all, agent_ids)

# Example inputs
new_args = dict()
new_args["input_dim"] = 18
new_args["num_heads"] = 4
new_args["num_agents"] = 3
new_args["num_policies"] = 8
new_args["embedding_dim"] = 14
new_args["output_dim"] = 5

obs_all = torch.randn(2, new_args["num_agents"], new_args["input_dim"])  # (batch_size, num_agents, obs_dim)
agent_ids = torch.arange(new_args["num_agents"]) # agent IDs

# Example model initialization
# input_dim + embedding_dim MUST be divisible by num_heads

# Create Agent objects for each agent
agents = []
agents.append(Agent(new_args))
new_args['model'] = agents[0].model
for i in range(1, new_args['num_agents']):
    agents.append(Agent(new_args))

# Assuming all agents use the same model, perform forward pass
for i, agent in enumerate(agents):
    actions = agent.select_action(obs_all, agent_ids)
    print(f'i:{i} actions.size():{actions.size()}')  # Expected size: (batch_size, action_dim)
    print(actions)  # Expected size: (batch_size, action_dim)
