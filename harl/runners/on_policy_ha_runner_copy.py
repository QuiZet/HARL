"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner_copy import OnPolicyBaseRunner
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic_copy import VCritic

class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def __init__(self, args, algo_args, env_args, agent_classes=None):
        print(f'Initializing OnPolicyHARunner with agent_classes: {agent_classes}')  # Debug print
        super().__init__(args, algo_args, env_args, agent_classes)
        print(f'self.agent_classes in HA runner: {self.agent_classes}')

    def train(self):
        """Train the model."""
        actor_train_infos = []
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        if self.use_class_ac:
            for class_id, agents in self.agent_classes.items():
                obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch = self.collect_class_batch(agents, advantages, factor)
                
                actor_train_info = self.class_actors[class_id].train(
                    (obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch)
                )
                actor_train_infos.append(actor_train_info)

                factor = self.update_factor(factor, class_id, agents)

            critic_train_info = []
            for class_id, critic in self.class_critics.items():
                critic_train_info.append(critic.train(self.critic_buffer, self.value_normalizer))
        else:
            for agent_id in agent_order:
                self.actor_buffer[agent_id].update_factor(
                    factor
                )  # current actor save factor

                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

                new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

                factor = factor * _t2n(
                    getattr(torch, self.action_aggregation)(
                        torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                    ).reshape(
                        self.algo_args["train"]["episode_length"],
                        self.algo_args["train"]["n_rollout_threads"],
                        1,
                    )
                )
                actor_train_infos.append(actor_train_info)
            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def collect_class_batch(self, agents, advantages, factor):
        obs_batch = []
        rnn_states_batch = []
        actions_batch = []
        masks_batch = []
        active_masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []
        available_actions_batch = []
        factor_batch = []

        for agent_id in agents:
            obs_batch.append(self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]))
            rnn_states_batch.append(self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]))
            actions_batch.append(self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]))
            masks_batch.append(self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]))
            active_masks_batch.append(self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]))
            old_action_log_probs_batch.append(check(self.actor_buffer[agent_id].old_action_log_probs[:-1]).to(**self.tpdv))
            adv_targ.append(check(advantages[:, :, agent_id].copy()).to(**self.tpdv))
            available_actions_batch.append(None if self.actor_buffer[agent_id].available_actions is None else self.actor_buffer[agent_id].available_actions[:-1].reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:]))
            factor_batch.append(check(factor).to(**self.tpdv))

        return torch.cat(obs_batch), torch.cat(rnn_states_batch), torch.cat(actions_batch), torch.cat(masks_batch), torch.cat(active_masks_batch), torch.cat(old_action_log_probs_batch), torch.cat(adv_targ), torch.cat(available_actions_batch), torch.cat(factor_batch)

    def update_factor(self, factor, class_id, agents):
        for agent_id in agents:
            available_actions = None if self.actor_buffer[agent_id].available_actions is None else self.actor_buffer[agent_id].available_actions[:-1].reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            old_actions_logprob, _, _ = self.class_actors[class_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:])
            )
            new_actions_logprob, _, _ = self.class_actors[class_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:])
            )
            factor = factor * _t2n(getattr(torch, self.action_aggregation)(torch.exp(new_actions_logprob - old_actions_logprob), dim=-1).reshape(self.algo_args['train']['episode_length'], self.algo_args['train']['n_rollout_threads'], 1))
        return factor
