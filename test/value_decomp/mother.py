import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_tag_v3
from collections import deque

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
CLIP_EPS = 0.2
EPOCHS = 3
BATCH_SIZE = 64
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 0.5
TIMESTEPS = 100000

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

# PPO Agent Class
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

# Function to collect experience
def collect_experience(env, adversary_agents, memory, episode_rewards):
    env.reset()
    prev_obs = {}
    prev_actions = {}
    prev_log_probs = {}
    prev_dones = {}
    prev_rewards = {}
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation
        if agent.startswith("adversary"):
            agent_idx = int(agent.split('_')[1])
            if agent in prev_obs:
                memory[agent_idx]['observations'].append(prev_obs[agent])
                memory[agent_idx]['actions'].append(prev_actions[agent])
                memory[agent_idx]['log_probs'].append(prev_log_probs[agent])
                memory[agent_idx]['rewards'].append(prev_rewards[agent])
                memory[agent_idx]['dones'].append(prev_dones[agent])
                episode_rewards[agent_idx] += prev_rewards[agent]
            if done:
                action = None
            else:
                # Get action for current observation
                action, log_prob, entropy = adversary_agents[agent_idx].get_action(observation)
                # Store current data for next step
                prev_obs[agent] = observation
                prev_actions[agent] = action
                prev_log_probs[agent] = log_prob
                prev_dones[agent] = done
            prev_rewards[agent] = reward
        else:
            if done:
                action = None
            else:
                action = env.action_space(agent).sample()
        env.step(action)
    # After the loop, handle the last step for each adversary
    for agent in env.agents:
        if agent.startswith("adversary") and agent in prev_obs:
            agent_idx = int(agent.split('_')[1])
            memory[agent_idx]['observations'].append(prev_obs[agent])
            memory[agent_idx]['actions'].append(prev_actions[agent])
            memory[agent_idx]['log_probs'].append(prev_log_probs[agent])
            memory[agent_idx]['rewards'].append(prev_rewards[agent])
            memory[agent_idx]['dones'].append(prev_dones[agent])
            episode_rewards[agent_idx] += prev_rewards[agent]

# Function to compute returns and advantages
def compute_gae(next_value, rewards, masks, values):
    values = np.append(values, next_value)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * gae * masks[step]
        returns.insert(0, gae + values[step])
    adv = np.array(returns) - values[:-1]
    return returns, adv

# Training Function
def train():
    env = simple_tag_v3.env()
    env.reset()

    # Identify adversaries
    adversary_ids = [agent for agent in env.possible_agents if agent.startswith("adversary")]
    num_adversaries = len(adversary_ids)

    # Initialize adversary agents
    adversary_agents = []
    for agent in adversary_ids:
        obs_size = env.observation_space(agent).shape[0]
        act_size = env.action_space(agent).n
        adversary_agents.append(PPOAgent(obs_size, act_size).to(device))

    # Initialize optimizers
    optimizers = [optim.Adam(agent.parameters(), lr=LR) for agent in adversary_agents]

    # Memory for each adversary
    memory = [{'observations': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []} for _ in range(num_adversaries)]

    # Initialize lists to store metrics
    total_rewards = [0 for _ in range(num_adversaries)]
    episode_rewards = [0 for _ in range(num_adversaries)]
    actor_losses = [[] for _ in range(num_adversaries)]
    critic_losses = [[] for _ in range(num_adversaries)]
    entropies = [[] for _ in range(num_adversaries)]

    timestep = 0
    episode = 0
    while timestep < TIMESTEPS:
        collect_experience(env, adversary_agents, memory, episode_rewards)
        timestep += 1

        # Update adversary agents
        for idx, agent in enumerate(adversary_agents):
            obs = torch.tensor(np.array(memory[idx]['observations']), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(memory[idx]['actions']), dtype=torch.long).to(device)
            old_log_probs = torch.stack(memory[idx]['log_probs']).detach().to(device)
            rewards = memory[idx]['rewards']
            dones = memory[idx]['dones']

            # Compute masks
            masks = [1 - int(done) for done in dones]

            # Compute returns and advantages
            with torch.no_grad():
                values = agent.get_value(obs)
                if values is None:
                    print("get_value returned None")
                    continue  # Skip this agent or handle appropriately
                values = values.cpu().numpy().flatten()
                if dones[-1]:
                    next_value = 0
                else:
                    next_obs = memory[idx]['observations'][-1]
                    next_value = agent.get_value(next_obs)
                    if next_value is None:
                        print("get_value returned None for next_obs")
                        next_value = 0
                    else:
                        next_value = next_value.item()
            returns, advantages = compute_gae(next_value, rewards, masks, values)

            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

            # PPO Update
            for _ in range(EPOCHS):
                # Shuffle the data
                indices = np.arange(len(rewards))
                np.random.shuffle(indices)
                for start in range(0, len(rewards), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    batch_indices = indices[start:end]

                    batch_obs = obs[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Get current policy outputs
                    probs = agent.actor(batch_obs)
                    dist = torch.distributions.Categorical(probs)
                    entropy = dist.entropy().mean()
                    new_log_probs = dist.log_prob(batch_actions)

                    # Ratio for PPO clipping
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)

                    # Surrogate loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = (agent.get_value(batch_obs).squeeze() - batch_returns).pow(2).mean()

                    loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * entropy

                    # Gradient descent step
                    optimizers[idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizers[idx].step()

                    # Collect metrics
                    actor_losses[idx].append(actor_loss.item())
                    critic_losses[idx].append(critic_loss.item())
                    entropies[idx].append(entropy.item())

            # Clear memory
            memory[idx] = {'observations': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

            # Update total rewards
            total_rewards[idx] += episode_rewards[idx]
            episode_rewards[idx] = 0  # Reset episode reward

        # Logging
        if timestep % 100 == 0:
            print(f"\nTimestep: {timestep}")
            for idx in range(num_adversaries):
                avg_actor_loss = np.mean(actor_losses[idx]) if actor_losses[idx] else 0
                avg_critic_loss = np.mean(critic_losses[idx]) if critic_losses[idx] else 0
                avg_entropy = np.mean(entropies[idx]) if entropies[idx] else 0
                avg_reward = total_rewards[idx] / (episode + 1) if episode + 1 > 0 else 0

                print(f"Adversary {idx}:")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Actor Loss: {avg_actor_loss:.4f}")
                print(f"  Average Critic Loss: {avg_critic_loss:.4f}")
                print(f"  Average Entropy: {avg_entropy:.4f}")

                # Clear metrics after logging
                actor_losses[idx] = []
                critic_losses[idx] = []
                entropies[idx] = []

            episode += 1

if __name__ == "__main__":
    train()
