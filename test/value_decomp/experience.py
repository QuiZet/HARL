# experience.py

from config import device
import torch

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
