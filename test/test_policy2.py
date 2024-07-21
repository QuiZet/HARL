import sys
sys.path.append('.')
import time
import torch
import torch.nn as nn
import torch.optim as optim

from harl.models.policy_models.embd_policy import EmbdPolicyNetwork

import gym
import numpy as np

class Agent:
    def __init__(self, args, device):
        self.model = EmbdPolicyNetwork(args, device).to(device)

    def select_action(self, obs_all, agent_ids):
        return self.model(obs_all, agent_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example inputs
input_dim_base_size = 6
new_args = {
    "num_heads": 4,
    "num_agents": 6,
    "num_policies": 8,
    "embedding_dim": 14,
    "output_dim": 5,
    "input_dim": input_dim_base_size,
    "obs_dim_resized": 18,
    "hidden_sizes": [128, 128],
    "activation_func": "relu",
    "final_activation_func": "tanh"
}
new_args['input_dim'] = new_args['input_dim'] * new_args['num_agents']

# # Define the dimensions for share_obs_space and act_space
# num_agents = 6
# obs_dim = 6 * num_agents
# act_dim = 5
# share_obs_dim = obs_dim * num_agents

# Create a Box space for share_obs_space and sample from it
share_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_args['input_dim'],), dtype=np.float32)
share_obs_sample = share_obs_space.sample()

# Create a list of Box spaces for act_space and sample from each
act_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(new_args['output_dim'],), dtype=np.float32) for _ in range(new_args['num_agents'])]
act_samples = [space.sample() for space in act_space]

new_args['obs_space'] = share_obs_space
new_args['action_space'] = act_space[0]

obs_all = torch.randn(2, new_args["num_agents"], new_args["input_dim"]).to(device)  # (batch_size, num_agents, obs_dim)
agent_ids = torch.arange(new_args["num_agents"]).to(device)  # agent IDs

# Create Agent objects for each agent
agents = [Agent(new_args, device=device)]
new_args['model'] = agents[0].model
for _ in range(1, new_args['num_agents']):
    agents.append(Agent(new_args, device))

# Define loss function and optimizer
criterion = nn.MSELoss()  # Example loss function, modify as needed
optimizer = optim.Adam(agents[0].model.parameters(), lr=1e-3)  # Example optimizer, modify as needed

num_epochs = 1000  # Example number of epochs

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    for i, agent in enumerate(agents):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        actions = agent.select_action(obs_all, agent_ids)
        
        # Example target, modify as needed
        #target = torch.zeros_like(actions)
        target = torch.randn(actions.shape).to(device)
        
        # Compute loss
        loss = criterion(actions, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print loss (optional)
        if (epoch + 1) % 1 == 0 and i == 0:  # Print every epoch for the first agent
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

exit(0)

end_time = time.time()
print(f'Training time: {end_time - start_time:.2f} seconds')

# Measure select_action computation time
for i, agent in enumerate(agents):
    action_start_time = time.time()
    actions = agent.select_action(obs_all, agent_ids)
    action_end_time = time.time()
    print(f'i:{i} actions.size():{actions.size()}')  # Expected size: (batch_size, action_dim)
    print(f'actions:{actions} device:{actions.device}')  # Expected size: (batch_size, action_dim)
    print(f'select_action computation time for agent {i}: {action_end_time - action_start_time:.6f} seconds')

# Save the entire model
model_path = 'embd_policy_network.pt'
torch.save(agents[0].model, model_path)
print(f'Model saved to {model_path}')

# Load the entire model
loaded_model = torch.load(model_path).to(device)
print(f'Model loaded from {model_path}')

# Verify the loaded model
for i, agent in enumerate(agents):
    actions = agent.select_action(obs_all, agent_ids)
    print(f'i:{i} actions.size():{actions.size()}')  # Expected size: (batch_size, action_dim)
    print(f'actions:{actions} device:{actions.device}')  # Expected size: (batch_size, action_dim)

# Change the number of agents, but keep the data size same
print('<><><><><><><><><><><><><><>')
print('Change the number of agents, but keep the data size same')
# Example inputs
new_args["num_agents"] = 7
new_args['input_dim'] = new_args['input_dim'] * new_args['num_agents']
agents.append(Agent(new_args, device))
obs_all = torch.randn(2, new_args["num_agents"], new_args["input_dim"]).to(device)  # (batch_size, num_agents, obs_dim)
agent_ids = torch.arange(new_args["num_agents"]).to(device)  # agent IDs
embedding_layer = nn.Embedding(num_embeddings=new_args["num_agents"], embedding_dim=new_args["embedding_dim"]).to(device)
# Verify the loaded model
for i, agent in enumerate(agents):
    agent.model.feature_extractor.embedding_layer = embedding_layer
    actions = agent.select_action(obs_all, agent_ids)
    print(f'i:{i} actions.size():{actions.size()}')  # Expected size: (batch_size, action_dim)
    print(f'actions:{actions} device:{actions.device}')  # Expected size: (batch_size, action_dim)



# TODO:
# Change the number of agents and observation dimension
# Few shot learning
# try how it works
# Check why it is not improving
