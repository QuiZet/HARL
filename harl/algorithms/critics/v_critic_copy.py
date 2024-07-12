"""V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.v_net import VNet


class VCritic:
    """V Critic.
    Critic that learns a V-function.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu"), use_class_ac=False, agent_classes=None):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_class_ac = use_class_ac
        self.agent_classes = agent_classes
        self.class_critics = {}

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.share_obs_space = cent_obs_space

        if self.use_class_ac:
            for class_id in agent_classes.keys():
                self.class_critics[class_id] = VNet(args, self.share_obs_space, self.device)
        else:
            self.critic = VNet(args, self.share_obs_space, self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.get_all_critic_parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_all_critic_parameters(self):
        """Get all parameters of class-specific or central critics."""
        if self.use_class_ac:
            params = []
            for critic in self.class_critics.values():
                params += list(critic.parameters())
            return params
        else:
            return self.critic.parameters()

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, rnn_states_critic, masks, class_id=None):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            class_id: (int) class ID for class-specific critics.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if self.use_class_ac:
            values, rnn_states_critic = self.class_critics[class_id](cent_obs, rnn_states_critic, masks)
        else:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, rnn_states_critic

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer=None
    ):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample, value_normalizer=None, class_id=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
            class_id: (int) class ID for class-specific critics.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _ = self.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch, class_id=class_id
        )

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.get_all_critic_parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.get_all_critic_parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        if self.use_class_ac:
            for class_id, agents in self.agent_classes.items():
                for _ in range(self.critic_epoch):
                    if self.use_recurrent_policy:
                        data_generator = critic_buffer[class_id].recurrent_generator_critic(
                            self.critic_num_mini_batch, self.data_chunk_length
                        )
                    elif self.use_naive_recurrent_policy:
                        data_generator = critic_buffer[class_id].naive_recurrent_generator_critic(
                            self.critic_num_mini_batch
                        )
                    else:
                        data_generator = critic_buffer[class_id].feed_forward_generator_critic(
                            self.critic_num_mini_batch
                        )

                    for sample in data_generator:
                        value_loss, critic_grad_norm = self.update(
                            sample, value_normalizer=value_normalizer, class_id=class_id
                        )

                        train_info["value_loss"] += value_loss.item()
                        train_info["critic_grad_norm"] += critic_grad_norm

            num_updates = self.critic_epoch * self.critic_num_mini_batch * len(self.agent_classes)
        else:
            for _ in range(self.critic_epoch):
                if self.use_recurrent_policy:
                    data_generator = critic_buffer.recurrent_generator_critic(
                        self.critic_num_mini_batch, self.data_chunk_length
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = critic_buffer.naive_recurrent_generator_critic(
                        self.critic_num_mini_batch
                    )
                else:
                    data_generator = critic_buffer.feed_forward_generator_critic(
                        self.critic_num_mini_batch
                    )

                for sample in data_generator:
                    value_loss, critic_grad_norm = self.update(
                        sample, value_normalizer=value_normalizer
                    )

                    train_info["value_loss"] += value_loss.item()
                    train_info["critic_grad_norm"] += critic_grad_norm

            num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        if self.use_class_ac:
            for critic in self.class_critics.values():
                critic.train()
        else:
            self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        if self.use_class_ac:
            for critic in self.class_critics.values():
                critic.eval()
        else:
            self.critic.eval()
