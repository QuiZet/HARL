"""CHATRPO algorithm."""

import numpy as np
import torch
from harl.utils.envs_tools import check
from harl.utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence,
)
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.stochastic_policy import StochasticPolicy


class CHATRPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize CHATRPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        assert (
            act_space.__class__.__name__ != "MultiDiscrete"
        ), "only continuous and discrete action space is supported by CHATRPO."
        super(CHATRPO, self).__init__(args, obs_space, act_space, device)

        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]

        self.use_class_ac = args.get('class_ac', True)

        if self.use_class_ac:
            self.class_actors = {}
            self.class_critics = {}
            self.class_id_map = {}

    def get_class_id_from_obs_act_space(self, obs_space, act_space):
        """Assign a unique class ID based on obs_space and act_space.
        Args:
            obs_space: (gym.spaces) observation space.
            act_space: (gym.spaces) action space.
        Returns:
            class_id: (int) class ID.
        """
        obs_act_tuple = (obs_space, act_space)
        if obs_act_tuple not in self.class_id_map:
            self.class_id_map[obs_act_tuple] = len(self.class_id_map) + 1
        return self.class_id_map[obs_act_tuple]

    def append_class_id_to_observation(self, observation, class_id):
        """Append class ID to the observation array or tensor.
        Args:
            observation: (np.ndarray or torch.Tensor) observation array or tensor.
            class_id: (int) class ID.
        Returns:
            observation_with_class_id: (np.ndarray or torch.Tensor) observation with class ID appended.
        """
        if isinstance(observation, np.ndarray):
            return np.append(observation, class_id)
        elif isinstance(observation, torch.Tensor):
            class_id_tensor = torch.tensor([class_id], dtype=observation.dtype, device=observation.device)
            return torch.cat((observation, class_id_tensor), dim=-1)
        else:
            raise TypeError("Unsupported observation type")

    def update(self, sample):
        """Update actor networks.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            kl: (torch.Tensor) KL divergence between old and new policy.
            loss_improve: (np.float32) loss improvement.
            expected_improve: (np.ndarray) expected loss improvement.
            dist_entropy: (torch.Tensor) action entropies.
            ratio: (torch.Tensor) ratio between new and old policy.
        """

        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        obs_space = obs_batch[0].shape
        act_space = actions_batch[0].shape
        class_id = self.get_class_id_from_obs_act_space(obs_space, act_space)
        obs_batch_with_class_id = [self.append_class_id_to_observation(obs, class_id) for obs in obs_batch]

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.class_actors[class_id].evaluate_actions(
            obs_batch_with_class_id,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        if self.use_policy_active_masks:
            loss = (
                torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ, dim=-1, keepdim=True
            ).mean()

        loss_grad = torch.autograd.grad(
            loss, self.class_actors[class_id].parameters(), allow_unused=True
        )
        loss_grad = flat_grad(loss_grad)

        step_dir = conjugate_gradient(
            self.class_actors[class_id],
            obs_batch_with_class_id,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            loss_grad.data,
            nsteps=10,
            device=self.device,
        )

        loss = loss.data.cpu().numpy()

        params = flat_params(self.class_actors[class_id])
        fvp = fisher_vector_product(
            self.class_actors[class_id],
            obs_batch_with_class_id,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            step_dir,
        )
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir

        old_actor = StochasticPolicy(
            self.args, self.obs_space, self.act_space, self.device
        )
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        # Backtracking line search (https://en.wikipedia.org/wiki/Backtracking_line_search)
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            update_model(self.class_actors[class_id], new_params)
            action_log_probs, dist_entropy, _ = self.class_actors[class_id].evaluate_actions(
                obs_batch_with_class_id,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )

            ratio = getattr(torch, self.action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1,
                keepdim=True,
            )
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(
                    ratio * factor_batch * adv_targ, dim=-1, keepdim=True
                ).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch_with_class_id,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                new_actor=self.class_actors[class_id],
                old_actor=old_actor,
            )
            kl = kl.mean()

            if (
                kl < self.kl_threshold
                and (loss_improve / expected_improve) > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.class_actors[class_id], params)
            print("policy update does not improve the surrogate")

        return kl, loss_improve, expected_improve, dist_entropy, ratio

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        for sample in data_generator:
            kl, loss_improve, expected_improve, dist_entropy, imp_weights = self.update(
                sample
            )

            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info