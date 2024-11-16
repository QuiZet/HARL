# train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_tag_v3

from config import *
from ppo_agent import PPOAgent
from critic import Critic
from decomposed_critic import DecomposedCritic
from experience import collect_experience
from gae import compute_gae
import random



# Configure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(use_decomposed_critic=False):
    env = simple_tag_v3.env()
    env.reset()

    # Identify adversaries
    adversary_ids = [agent for agent in env.possible_agents if agent.startswith("adversary")]
    num_adversaries = len(adversary_ids)

    # Initialize adversary agents
    adversary_agents = []
    obs_sizes = []
    act_sizes = []
    for agent_id in adversary_ids:
        obs_size = env.observation_space(agent_id).shape[0]
        act_size = env.action_space(agent_id).n
        obs_sizes.append(obs_size)
        act_sizes.append(act_size)
        adversary_agents.append(PPOAgent(obs_size, act_size).to(device))

    # Initialize critics
    if use_decomposed_critic:
        critic = DecomposedCritic()  # Placeholder
    else:
        critics = [Critic(obs_size).to(device) for obs_size in obs_sizes]

    # Initialize optimizers
    if use_decomposed_critic:
        optimizers = [optim.Adam(agent.parameters(), lr=LR) for agent in adversary_agents]
        critic_optimizer = optim.Adam(critic.parameters(), lr=LR)
    else:
        optimizers = [optim.Adam(list(agent.parameters()) + list(critic.parameters()), lr=LR)
                      for agent, critic in zip(adversary_agents, critics)]

    # Memory for each adversary
    memory = [{'observations': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
              for _ in range(num_adversaries)]

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
                if use_decomposed_critic:
                    # Placeholder: Implement decomposed critic logic here
                    values = np.zeros(len(rewards))
                    next_value = 0
                else:
                    values = critics[idx](obs).cpu().numpy().flatten()
                    if dones[-1]:
                        next_value = 0
                    else:
                        next_obs = memory[idx]['observations'][-1]
                        next_value = critics[idx](next_obs).item()
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

                    # Critic loss
                    if use_decomposed_critic:
                        # Placeholder: Implement decomposed critic loss computation
                        critic_loss = torch.tensor(0.0)  # Dummy value
                    else:
                        value_preds = critics[idx](batch_obs).squeeze()
                        critic_loss = (value_preds - batch_returns).pow(2).mean()

                    loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * entropy

                    # Gradient descent step
                    optimizers[idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    if not use_decomposed_critic:
                        nn.utils.clip_grad_norm_(critics[idx].parameters(), MAX_GRAD_NORM)
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
