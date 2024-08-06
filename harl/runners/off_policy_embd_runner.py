"""Runner for off-policy HARL algorithms."""
import time
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.off_policy_embd_base_runner import OffPolicyEmbdBaseRunner
from einops import rearrange

class OffPolicyEmbdRunner(OffPolicyEmbdBaseRunner):
    """Runner for off-policy Embedded algorithms."""

    def train(self, debug=False):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        
        # Train critic
        critic_start_time = time.time()
        self.critic.turn_on_grad()
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            agent_ids = torch.arange(self.num_agents)
            agent_ids_expanded = agent_ids.unsqueeze(0).expand(sp_next_obs.shape[1], -1)  # (batch_size, num_agents)
            # Swap dimensions using einops
            tensor_swapped = rearrange(sp_next_obs, 'a b c -> b a c')
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id], tensor_swapped, agent_ids_expanded)
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        critic_end_time = time.time()
        critic_time = critic_end_time - critic_start_time
        if debug:
            print(f"Critic training took {critic_time:.6f} seconds")
        
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # Train actors
            actor_start_time = time.time()
            if self.args["algo"] == "hasac":
                actions = []
                logp_actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        action, logp_action = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # Train this agent
                    actions[agent_id], logp_actions[agent_id] = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * sp_valid_transition[agent_id]
                                )
                                / sp_valid_transition[agent_id].sum()
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # Train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # Train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # Actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # Critic preds
                        critic_values = get_values()
                        # Update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    agent_ids = torch.arange(self.num_agents)
                    agent_ids_expanded = agent_ids.unsqueeze(0).expand(sp_next_obs.shape[1], -1)  # (batch_size, num_agents)
                    # Swap dimensions using einops
                    tensor_swapped = rearrange(sp_next_obs, 'a b c -> b a c')
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False, tensor_swapped, agent_ids_expanded
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    #agent_ids = torch.arange(self.num_agents) # error, it does not preserve the permutation
                    agent_ids = torch.as_tensor(agent_order)
                    agent_ids_expanded = agent_ids.unsqueeze(0).expand(sp_next_obs.shape[1], -1)  # (batch_size, num_agents)

                    # Ensure agent_ids are within the range of the second dimension of obs_all
                    #print(f'torch.max(agent_ids_expanded):{torch.max(agent_ids_expanded)}')  # Should be torch.Size([1000, 3, 18])
                    #print(f'sp_obs.size(1):{sp_obs.shape} tensor_swapped:{tensor_swapped.shape}')  # Should be torch.Size([1000, 3, 18])
                    assert torch.max(agent_ids_expanded) < tensor_swapped.shape[1], "agent_ids contain indices out of range for obs_all"
                    #print(f'agent_ids_expanded:{agent_ids_expanded.shape}')  # Should be torch.Size([1000, 3, 18])
                    tensor_swapped = torch.Tensor(tensor_swapped).to(agent_ids_expanded.device)
                    #print(f'tensor_swapped:{type(tensor_swapped)} agent_ids_expanded:{type(agent_ids_expanded)}')
                    # Reorder obs_all based on agent_ids
                    agent_ids_expanded_gather = agent_ids_expanded.unsqueeze(-1).expand(-1, -1, tensor_swapped.shape[2])
                    #print(f'agent_ids_expanded:{agent_ids_expanded.shape}')  # Should be torch.Size([1000, 3, 18])
                    reordered_tensor_swapped = tensor_swapped.gather(1, agent_ids_expanded_gather)
                    #print(f'sp_share_obs{sp_share_obs.shape} reordered_obs_all:{reordered_obs_all.shape}')  # Should be torch.Size([1000, 3, 18])
                    #print(f'agent_ids_expanded{agent_ids_expanded}')
                    #print(f'tensor_swapped{tensor_swapped} reordered_obs_all:{reordered_obs_all}')  # Should be torch.Size([1000, 3, 18])

                    for agent_id in agent_order:
                        #print(f'agent_id:{agent_id}')
                        #print(f'sp_obs{sp_obs.shape} reordered_tensor_swapped:{reordered_tensor_swapped.shape} agent_ids_expanded:{agent_ids_expanded.shape}')  # Should be torch.Size([1000, 3, 18])

                        self.actor[agent_id].turn_on_grad()
                        # Train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                        #    sp_obs[agent_id], False, reordered_tensor_swapped, agent_ids_expanded
                            reordered_tensor_swapped[:, agent_id, :], False, reordered_tensor_swapped, agent_ids_expanded
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t, agent_ids_expanded)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()

                        #if self.total_it % (2*self.policy_freq) == 0:
                        #    print(f'A agent_id[{agent_id}] actions:{actions[agent_id]} actor_loss:{actor_loss}')

                        actions[agent_id] = self.actor[agent_id].get_actions(
                        #    sp_obs[agent_id], False, reordered_tensor_swapped, agent_ids_expanded
                            reordered_tensor_swapped[:, agent_id, :], False, reordered_tensor_swapped, agent_ids_expanded
                        )

                        #if self.total_it % (2*self.policy_freq) == 0:
                        #    print(f'B agent_id[{agent_id}] actions:{actions[agent_id]} actor_loss:{actor_loss}')


                # Soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
            actor_end_time = time.time()
            actor_time = actor_end_time - actor_start_time
            if debug:
                print(f"Actor training took {actor_time:.6f} seconds")
