import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner_copy import OnPolicyBaseRunner

class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def __init__(self, args, algo_args, env_args, agent_classes=None):
        print(f'Initializing OnPolicyHARunner with agent_classes: {agent_classes}')  # Debug print
        super().__init__(args, algo_args, env_args, agent_classes)
        print(f'self.agent_classes in HA runner: {self.agent_classes}')

    def train(self):
        """Train the model."""
        print(f'at the start of train method of OnPolicyHARunner self.agent_classes is {self.agent_classes}')  # Debug print
        actor_train_infos = []
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        advantages = {}
        if self.value_normalizer is not None:
            print(f'using value_normalizer in train method of OnPolicyHARunner')  # Debug print
            for class_key, class_id in self.agent_classes.items():
                if isinstance(class_id, int):  # Skip the string keys
                    buffer = self.critic_buffer[class_id]
                    advantages[class_id] = buffer.returns[
                        :-1
                    ] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            print(f'not using value_normalizer in train method of OnPolicyHARunner')  # Debug print
            for class_key, class_id in self.agent_classes.items():
                if isinstance(class_id, int):  # Skip the string keys
                    buffer = self.critic_buffer[class_id]
                    advantages[class_id] = buffer.returns[:-1] - buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[class_id].active_masks for class_id in self.agent_classes.values() if isinstance(class_id, int)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = {class_id: adv.copy() for class_id, adv in advantages.items()}
            for class_id in advantages_copy.keys():
                advantages_copy[class_id][active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(np.concatenate([adv for adv in advantages_copy.values()]))
            std_advantages = np.nanstd(np.concatenate([adv for adv in advantages_copy.values()]))
            for class_id in advantages.keys():
                advantages[class_id] = (advantages[class_id] - mean_advantages) / (std_advantages + 1e-5)

        if self.fixed_order:
            print('using fixed_order block of train method of OnPolicyHARunner')  # Debug print
            agent_order = list(range(self.num_agents))
        else:
            print('using random_order block of train method of OnPolicyHARunner')  # Debug print
            agent_order = list(torch.randperm(self.num_agents).numpy())
            
        print(f'in use_class_ac block of train method of OnPolicyHARunner, advantages is {advantages}') # Debug print
        
        if self.use_class_ac:
            print(f'self.agent_classes in train method of OnPolicyHARunner: {self.agent_classes}')  # Debug print
            for class_key, class_id in self.agent_classes.items():
                if isinstance(class_id, str):
                    continue
                print(f'class_id in self.agent_classes.items() in train method of OnPolicyHARunner: {class_id}')  # Debug print
                if class_id not in advantages:
                    print(f"Warning: advantages is {advantages} class_id {class_id} not found in advantages. Skipping...")
                    continue
                print(f'advantages used for class_id {class_id} in train method of OnPolicyHARunner: {advantages[class_id]}')  # Debug print

                actor_buffers = [self.actor_buffer[agent_id] for agent_id in self.agent_classes[class_id]]
                actor_train_info = self.class_actors[class_id].train(actor_buffers, advantages[class_id], self.state_type)
                
                print('entering actor train block of train method of OnPolicyHARunner')  # Debug print
                
                actor_train_infos.append(actor_train_info)
                print('finished appending information to actor_train_infos')  # Debug print

                factor = self.update_factor(factor, class_id, self.agent_classes[class_id])
                print('exiting update_factor block of train method of OnPolicyHARunner')  # Debug print

            critic_train_info = []
            for class_id, critic in self.class_critics.items():
                critic_train_info.append(critic.train(self.critic_buffer[class_id], self.value_normalizer))
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

                actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
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
                        torch.exp(new_actions_logprob - actions_logprob), dim=-1
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
        action_log_probs_batch = []
        adv_targ = []
        available_actions_batch = []
        factor_batch = []

        for agent_id in agents:
            class_id = self.get_class_id(agent_id)
            
            obs_batch.append(torch.tensor(self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:])))
            rnn_states_batch.append(torch.tensor(self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:])))
            actions_batch.append(torch.tensor(self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:])))
            masks_batch.append(torch.tensor(self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:])))
            active_masks_batch.append(torch.tensor(self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:])))
            action_log_probs_batch.append(torch.tensor(self.actor_buffer[agent_id].action_log_probs[:-1]))
            adv_targ.append(torch.tensor(advantages[:, :, agent_id].copy()))
            available_actions_batch.append(None if self.actor_buffer[agent_id].available_actions is None else torch.tensor(self.actor_buffer[agent_id].available_actions[:-1].reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])))
            factor_batch.append(torch.tensor(factor))

        return (
            torch.cat(obs_batch),
            torch.cat(rnn_states_batch),
            torch.cat(actions_batch),
            torch.cat(masks_batch),
            torch.cat(active_masks_batch),
            torch.cat(action_log_probs_batch),
            torch.cat(adv_targ),
            torch.cat(available_actions_batch),
            torch.cat(factor_batch)
        )

    def update_factor(self, factor, class_id, agents):
        for agent_id in agents:
            available_actions = None if self.actor_buffer[agent_id].available_actions is None else self.actor_buffer[agent_id].available_actions[:-1].reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            actions_logprob, _, _ = self.class_actors[class_id].evaluate_actions(
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
            factor = factor * _t2n(getattr(torch, self.action_aggregation)(torch.exp(new_actions_logprob - actions_logprob), dim=-1).reshape(self.algo_args['train']['episode_length'], self.algo_args['train']['n_rollout_threads'], 1))
        return factor

    def get_class_id(self, agent_id):
        print(f'agent id in get_class_id: {agent_id}')  # Debug print
        for class_key, class_value in self.agent_classes.items():
            if isinstance(class_value, list):  # Ensure we are only checking lists of agents
                print(f'self.agent_classes.items() in get_class_id: {self.agent_classes.items()}')  # Debug print
                print(f'agents in get_class_id: {class_value}')  # Debug print
                if agent_id in class_value:
                    return class_key
        raise KeyError(f"Agent ID {agent_id} not found in any class")
