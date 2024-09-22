"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_chatrpo_base_runner import OnPolicyCHATRPOBaseRunner

class OnPolicyCHATRPORunner(OnPolicyCHATRPOBaseRunner):
    """Runner for on-policy HARL algorithms."""

    def __init__(self, args, algo_args, env_args):
        # Call the base class constructor
        super(OnPolicyCHATRPORunner, self).__init__(args, algo_args, env_args)
        
        # After initialization, print to check if agent_classes is correctly inherited
        print(f'self.agent_classes in __init__ of OnPolicyCHATRPORunner is : {self.agent_classes}')

    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(len(self.agent_classes))
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        class_order = list(self.agent_classes.keys())
        if not self.fixed_order:
            class_order = list(torch.randperm(len(class_order)).numpy())
            class_order = [list(self.agent_classes.keys())[i] for i in class_order]

        print(f'class_order:{class_order} type:{type(class_order)}')
        for class_label in class_order:
            class_info = self.agent_classes[class_label]
            class_actor = self.class_actors[class_label]
            for agent_id in class_info[f"class_{class_label}_agents"]:
                self.actor_buffer[agent_id].update_factor(factor)

                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                old_actions_logprob, _, _ = class_actor.evaluate_actions(
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
                    actor_train_info = class_actor.train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = class_actor.train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

                new_actions_logprob, _, _ = class_actor.evaluate_actions(
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
                #print(f'factor is : {factor}')
                actor_train_infos.append(actor_train_info)
            print(f'new actions is: {new_actions_logprob.shape}')
            print(f'self.actor_buffer[int(class_label)].obs[:-1]: {self.actor_buffer[agent_id].obs[:-1].shape}')

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
    
    
    def train_new(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
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

        class_order = list(self.agent_classes.keys())
        #print(f'class_order:{class_order} type:{type(class_order)}')
        if not self.fixed_order:
            class_order = list(torch.randperm(len(class_order)).numpy())
            class_order = [list(self.agent_classes.keys())[i] for i in class_order]

        #print(f'class_order:{class_order} type:{type(class_order)}')
        # for class_label in class_order:
        #     class_info = self.agent_classes[class_label]
        #     class_actor = self.class_actors[class_label]
        #     for agent_id in class_info[f"class_{class_label}_agents"]:
        #         print(f'agent_id:{agent_id} class:{class_label}')

        for class_label in class_order:
            for agent_id in agent_order:
                # print information
                if self.agent_id_class[agent_id] is not class_label:
                    continue
                #print(f'class_label:{class_label} agent_id:{agent_id} class:{self.agent_id_class[agent_id]}')
                class_actor = self.class_actors[class_label]

                self.actor_buffer[agent_id].update_factor(
                    factor
                )  # current actor save factor

                # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                # compute action log probs for the actor before update.
                #old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                old_actions_logprob, _, _ = class_actor.evaluate_actions(
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

                # update actor
                if self.state_type == "EP":
                    #actor_train_info = self.actor[agent_id].train(
                    actor_train_info = class_actor.train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    #actor_train_info = self.actor[agent_id].train(
                    actor_train_info = class_actor.train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

                # compute action log probs for updated agent
                #new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                new_actions_logprob, _, _ = class_actor.evaluate_actions(
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

                # update factor for next agent
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

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
