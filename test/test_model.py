import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

from harl.models.policy_models.embd_policy import EmbdPolicyNetwork
from harl.models.policy_models.deterministic_policy import DeterministicPolicy

import gym
import numpy as np

import pandas as pd

class Agent:
    def __init__(self, args, device):
        print(f'Agent args:{args}')
        self.model = EmbdPolicyNetwork(args, device).to(device)
        print(f'self.model:{self.model}')

    def select_action(self, obs_all, agent_ids):
        return self.model(obs_all, agent_ids)

    def initialize_layers(self, new_obs_dim, new_num_agents, freeze_existing=True):
        # Initialize new layers or modify existing ones here
        # This is a placeholder for actual initialization logic
        pass

try:
    import torch.cuda.amp as amp

    def train_model(agents, actors, data_loader, criterion, optimizer, scheduler, n_epochs, device):
        training_start_time = time.time()
        
        scaler = amp.GradScaler()

        # Loss functions
        criterion_hard = nn.CrossEntropyLoss()
        criterion_soft = nn.KLDivLoss(reduction='batchmean')
        alpha = 0.1
        temperature = 0.5

        # how often print training information
        epochs_print = 100

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            for obs, actions, agent_ids, targets in data_loader:
                for agent_id, agent in enumerate(agents):
                    optimizer.zero_grad()
                    
                    with amp.autocast():
                        agent_obs = obs[:, :, :].to(device)
                        agent_ids_batch = agent_ids[:, :].to(device)
                        student_outputs = agent.model(agent_obs, agent_ids_batch)
                        # Fit or learn from another model
                        if actors is None:
                            loss = criterion(student_outputs, targets.to(device))
                            if epoch % epochs_print == 0:
                                print(f'student:{student_outputs[0]}')
                        else:
                            teacher_outputs = actors[agent_id](agent_obs[:, agent_id].to(device))
                            # # Soft targets
                            # soft_targets = torch.softmax(teacher_outputs / temperature, dim=1)
                            # # Calculate losses
                            # hard_loss = criterion_hard(student_outputs, targets.to(device))
                            # soft_loss = criterion_soft(torch.log_softmax(student_outputs / temperature, dim=1), soft_targets)
                            # loss = hard_loss + alpha * soft_loss  # Combine losses with weight alpha

                            # Output clipping
                            #student_outputs = torch.clamp(student_outputs, min=0.0, max=1.0)

                            loss = criterion(student_outputs, teacher_outputs)
                            if epoch % epochs_print == 0:
                                print(f'agent_id:{agent_id} teacher:{teacher_outputs[0]} student:{student_outputs[0]}')

                        if torch.isnan(loss) or torch.isinf(loss):
                            print("NaN or Inf detected in loss!")
                            continue

                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN or Inf detected in gradients!")
                        continue
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}, Epoch Time: {epoch_time:.4f}s")
        
        total_training_time = time.time() - training_start_time
        print(f"Total Training Time: {total_training_time:.4f}s")

except:
    def train_model(agents, data_loader, criterion, optimizer, scheduler, n_epochs):
        training_start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            for obs, actions, agent_ids, targets in data_loader:
                for agent_id, agent in enumerate(agents):
                    optimizer.zero_grad()
                    
                    forward_start_time = time.time()
                    # agent_obs = obs[:, agent_id, :]
                    # agent_ids_batch = agent_ids[:, agent_id]
                    agent_obs = obs[:, :, :]
                    agent_ids_batch = agent_ids[:, :]
                    # print(f'obs:{obs.shape}')
                    # print(f'agent_ids:{agent_ids.shape}')
                    # print(f'agent_obs:{agent_obs.shape}')
                    # print(f'agent_ids_batch:{agent_ids_batch.shape}')
                    outputs = agent.model(agent_obs, agent_ids_batch)
                    forward_time = time.time() - forward_start_time
                    
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}, Epoch Time: {epoch_time:.4f}s, Forward Time: {forward_time:.4f}s")
        
        total_training_time = time.time() - training_start_time
        print(f"Total Training Time: {total_training_time:.4f}s")


def create_data_loader(obs, actions, agent_ids, targets, batch_size):
    dataset = TensorDataset(obs, actions, agent_ids, targets)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return data_loader

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example inputs
input_dim_base_size = 6
new_args = {
    "num_heads": 4,
    "num_agents": 3,
    "num_policies": 4,
    "embedding_dim": 14,
    "output_dim": 5,
    "input_dim": input_dim_base_size,
    "obs_dim_resized": 18,
    "hidden_sizes": [128, 128],
    "activation_func": "relu",
    "final_activation_func": "tanh"
}
new_args['input_dim'] = new_args['input_dim'] * new_args['num_agents']
# Create a Box space for share_obs_space and sample from it
share_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_args['input_dim'],), dtype=np.float32)
share_obs_sample = share_obs_space.sample()
# Create a list of Box spaces for act_space and sample from each
act_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(new_args['output_dim'],), dtype=np.float32) for _ in range(new_args['num_agents'])]
act_samples = [space.sample() for space in act_space]
new_args['obs_space'] = share_obs_space
new_args['action_space'] = act_space[0]

# Initial dimensions
obs_dim = new_args['input_dim']
action_dim = new_args['output_dim']
num_agents = new_args['num_agents']
attention_heads = new_args['num_heads']
batch_size = 512 #32
n_epochs = 1000
#save_path = 'FeatureExtractor_model.pth'

# Create Agent objects for each agent
agents = [Agent(new_args, device=device) for _ in range(new_args['num_agents'])]


# # Example data
# num_samples = 10000
# obs = torch.randn((num_samples, num_agents, obs_dim))
# actions = torch.randn((num_samples, num_agents, action_dim))
# agent_ids = torch.arange(num_agents)
# agent_ids_expanded = agent_ids.unsqueeze(0).expand(num_samples, -1)  # (batch_size, num_agents)
# targets = torch.randn((num_samples, new_args['output_dim']))


# # Initialize lists to collect data
# obs_list = []
# actions_list = []
# target_actions_list = []

# # Load data from CSV files
# file_pattern = 'haddpg_data_*.csv'
# for idx in range(1,4):
#     file = f'haddpg_data_{idx}.csv'
#     df = pd.read_csv(file)
#     obs_list.append(df['obs'].apply(parse_list_string).tolist())
#     actions_list.append(df['actions'].apply(parse_list_string).tolist())
#     target_actions_list.append(df['target_actions'].apply(parse_list_string).tolist())

# # Convert lists to tensors with extra dimension for agents
# obs = torch.tensor(obs_list, dtype=torch.float32).transpose(0, 1)
# actions = torch.tensor(actions_list, dtype=torch.float32).transpose(0, 1)
# target_actions = torch.tensor(target_actions_list, dtype=torch.float32).transpose(0, 1)

# # obs_list = df['obs'].apply(parse_list_string).tolist()
# # actions_list = df['actions'].apply(parse_list_string).tolist()
# # target_actions_list = df['target_actions'].apply(parse_list_string).tolist()

# # Convert lists to tensors
# obs = torch.tensor(obs_list, dtype=torch.float32)
# actions = torch.tensor(actions_list, dtype=torch.float32)
# target_actions = torch.tensor(target_actions_list, dtype=torch.float32)

# print(f'obs:{obs.shape} actions:{actions.shape}')


import ast
import re
import csv
def parse_list_string(x):
    try:
        # Convert string to list safely
        return ast.literal_eval(x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing string: {x}\n{e}")
        return []

def read_custom_csv(file_path):
    obs_list = []
    actions_list = []
    target_actions_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process header
    header = lines[0].strip()
    print(f"Header: {header}")

    for line in lines[1:]:
        line = line.strip()
        if line:
            # Manually split the line into three parts based on ',[' and ']'
            parts = line.split('],[')
            if len(parts) == 3:
                obs_str = parts[0].strip('[]')
                actions_str = parts[1].strip('[]')
                target_actions_str = parts[2].strip('[]')
                
                # Ensure that the trailing ']' is removed from target_actions_str
                target_actions_str = target_actions_str.rstrip(']')
                
                obs = parse_list_string(f'[{obs_str}]')
                actions = parse_list_string(f'[{actions_str}]')
                target_actions = parse_list_string(f'[{target_actions_str}]')

                # Debug information
                print(f"Parsed obs: {obs}")
                print(f"Parsed actions: {actions}")
                print(f"Parsed target_actions: {target_actions}")

                if len(obs) > 0 and len(actions) > 0 and len(target_actions) > 0:
                    obs_list.append(obs)
                    actions_list.append(actions)
                    target_actions_list.append(target_actions)
            else:
                print(f"Skipping malformed line: {line}")

    return obs_list, actions_list, target_actions_list

obs_list_all = []
actions_list_all = []
target_actions_all = []

for idx in range(1, 4):
    # File path
    file_path = f'haddpg_data_{idx}.csv'  # Replace with your file path

    # Read the data
    obs_list, actions_list, target_actions_list = read_custom_csv(file_path)

    # Convert lists to tensors
    obs = torch.tensor(obs_list, dtype=torch.float32)
    actions = torch.tensor(actions_list, dtype=torch.float32)
    target_actions = torch.tensor(target_actions_list, dtype=torch.float32)
    obs_list_all.append(obs)
    actions_list_all.append(actions)
    target_actions_all.append(target_actions)

obs = torch.stack(obs_list_all, dim=1)
actions = torch.stack(actions_list_all, dim=1)
target_actions = torch.stack(target_actions_all, dim=1)
    
# Check dimensions
print("Shape of obs:", obs.shape)
print("Shape of actions:", actions.shape)
print("Shape of target_actions:", target_actions.shape)

# Check if actions tensor is of expected shape
expected_shape = (250, 5)  # Replace with the actual expected shape
if actions.shape != expected_shape:
    print(f"Unexpected shape for actions: {actions.shape}")


# # Example target data (adjust according to your needs)
# targets = torch.randn((len(obs_list), new_args['output_dim']))

targets = actions


num_samples = obs.shape[0]
agent_ids = torch.arange(num_agents)
agent_ids_expanded = agent_ids.unsqueeze(0).expand(num_samples, -1)  # (batch_size, num_agents)
# Create data loader
data_loader = create_data_loader(obs, actions, agent_ids_expanded, targets, batch_size)


# Load an existing model
actors = None
if True:
    actors = []
    for i in range(3):
        id = i
        actor = DeterministicPolicy(new_args, share_obs_space, act_space[0], device=device) # no initialization
        actor_state_dict = torch.load(str('/home/moro/workspace/university/todai/Simon/HARL/results/pettingzoo_mpe/simple_spread_v2-continuous/hatd3/pzhaddpg/seed-00001-2024-07-05-13-07-59/models') + "/actor_agent" + str(id) + ".pt")
        actor.load_state_dict(actor_state_dict)
        actor = actor.to(device)
        print(f'actor scale:{actor.scale} mean:{actor.mean}')
        print(f'agent scale:{agents[0].model.scale} mean:{agents[0].model.mean}')
        actors.append(actor)


# Define loss and optimizer
criterion = nn.MSELoss()
if actor is None:
    optimizer = optim.AdamW([param for agent in agents for param in agent.model.parameters()], lr=0.001)
else:
    optimizer = optim.AdamW([param for agent in agents for param in agent.model.parameters()], lr=0.001)
# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)





# Train the model
#train_model(agents, data_loader, criterion, optimizer, scheduler, n_epochs)

# Profiling the model training
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#    with record_function("model_training"):
        # Train the model
train_model(agents, actors, data_loader, criterion, optimizer, scheduler, n_epochs, device)

# Save the model after initial training
for idx, agent in enumerate(agents):
    save_model(agent.model, f'/home/moro/workspace/university/todai/Simon/HARL/results/seed-00001-2024-07-18-16-38-12/models/actor_agent{idx}.pt')
    #save_model(agent.model, f'/home/moro/workspace/university/todai/Simon/HARL/results/seed-00001-2024-07-18-16-38-12/models/target_actor_agent{idx}.pt')

# Print profiling results
#print(prof.key_averages().table(sort_by="cuda_time_total"))

# todo
exit(0)

# Load the model for each agent
loaded_agents = [Agent(new_args, device=device) for _ in range(new_args['num_agents'])]
for idx, agent in enumerate(loaded_agents):
    agent.model = load_model(agent.model, f'{save_path}_agent_{idx}')

# New dimensions after the first iteration
new_obs_dim = 18
new_num_agents = 6
n_epochs = 1000

# Modify the loaded models with new dimensions
for agent in loaded_agents:
    agent.initialize_layers(new_obs_dim, new_num_agents, freeze_existing=True)

# Example data
num_samples = 20
new_obs = torch.randn((num_samples, new_num_agents, new_obs_dim))
new_actions = torch.randn((num_samples, new_num_agents, action_dim))
new_agent_ids = torch.arange(new_num_agents)
new_agent_ids_expanded = new_agent_ids.unsqueeze(0).expand(num_samples, -1)  # (batch_size, num_agents)
new_targets = torch.randn((num_samples, new_args['output_dim']))

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW([param for agent in loaded_agents for param in agent.model.parameters()], lr=0.001)
# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

# Create new data loader for few-shot learning
new_data_loader = create_data_loader(new_obs, new_actions, new_agent_ids_expanded, new_targets, batch_size)

# Few-shot learning with the modified models
train_model(loaded_agents, new_data_loader, criterion, optimizer, scheduler, n_epochs=n_epochs)
