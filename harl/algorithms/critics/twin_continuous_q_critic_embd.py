"""Twin Continuous Q Critic."""
import itertools
from copy import deepcopy
import torch
from harl.models.value_function_models.embd_value import EmbdValueNetwork
from harl.models.value_function_models.continuous_q_net import ContinuousQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class TwinContinuousQCriticEmbd:
    """Twin Continuous Q Critic.
    Critic that learns two Q-functions. The action space is continuous.
    Note that the name TwinContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__

        new_args = dict()
        new_args["num_policies"] = 4
        new_args["num_heads"] = 4
        new_args["num_agents"] = num_agents
        new_args["embedding_dim"] = 14 #9
        new_args["output_dim"] = 1
        new_args["obs_dim_resized"] = 18

        merged_dict = args.copy()
        merged_dict.update(new_args)
        print(f'merged_dict:{merged_dict}')
       
        print(f'share_obs_space >> :{share_obs_space} act_space:{act_space} <<')
        self.critic = EmbdValueNetwork(merged_dict, share_obs_space, act_space, device)
        self.critic2 = EmbdValueNetwork(merged_dict, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def get_values(self, share_obs, actions, agent_ids):
        """Get the Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.critic(share_obs, actions, agent_ids.to(share_obs.device))

    # def train(
    #     self,
    #     share_obs,
    #     actions,
    #     reward,
    #     done,
    #     term,
    #     next_share_obs,
    #     next_actions,
    #     gamma,
    # ):
    #     """Train the critic.
    #     Args:
    #         share_obs: (np.ndarray) shape is (batch_size, dim)
    #         actions: (np.ndarray) shape is (n_agents, batch_size, dim)
    #         reward: (np.ndarray) shape is (batch_size, 1)
    #         done: (np.ndarray) shape is (batch_size, 1)
    #         term: (np.ndarray) shape is (batch_size, 1)
    #         next_share_obs: (np.ndarray) shape is (batch_size, dim)
    #         next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
    #         gamma: (np.ndarray) shape is (batch_size, 1)
    #     """
    #     assert share_obs.__class__.__name__ == "ndarray"
    #     assert actions.__class__.__name__ == "ndarray"
    #     assert reward.__class__.__name__ == "ndarray"
    #     assert done.__class__.__name__ == "ndarray"
    #     assert term.__class__.__name__ == "ndarray"
    #     assert next_share_obs.__class__.__name__ == "ndarray"
    #     assert gamma.__class__.__name__ == "ndarray"
    #     share_obs = check(share_obs).to(**self.tpdv)
    #     actions = check(actions).to(**self.tpdv)
    #     actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
    #     reward = check(reward).to(**self.tpdv)
    #     done = check(done).to(**self.tpdv)
    #     term = check(term).to(**self.tpdv)
    #     gamma = check(gamma).to(**self.tpdv)
    #     next_share_obs = check(next_share_obs).to(**self.tpdv)

    #     #print(f'CriticEmbd::train share_obs:{share_obs.shape} actions:{actions.shape}')
    #     agent_ids = torch.arange(self.num_agents).to(share_obs.device)
    #     agent_ids_expanded = agent_ids.unsqueeze(0).expand(share_obs.shape[0], -1)  # (batch_size, num_agents)
    #     #print(f'CriticEmbd::train agent_ids_expanded:{agent_ids_expanded.shape}')

    #     next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
    #     print('target_critic')
    #     next_q_values1 = self.target_critic(next_share_obs, next_actions, agent_ids_expanded)
    #     print('target_critic2')
    #     next_q_values2 = self.target_critic2(next_share_obs, next_actions, agent_ids_expanded)
    #     #print(f'CriticEmbd::target_critic next_actions:{next_share_obs.shape} next_actions:{next_actions.shape}')
    #     #print(f'CriticEmbd::next_q_values1:{next_q_values1.shape} next_q_values2:{next_q_values2.shape}')
    #     next_q_values = torch.min(next_q_values1, next_q_values2)
    #     if self.use_proper_time_limits:
    #         q_targets = reward + gamma * next_q_values * (1 - term)
    #     else:
    #         q_targets = reward + gamma * next_q_values * (1 - done)
    #     print('critic')
    #     critic_loss1 = torch.mean(
    #         torch.nn.functional.mse_loss(self.critic(share_obs, actions, agent_ids_expanded), q_targets)
    #     )
    #     print('critic2')
    #     critic_loss2 = torch.mean(
    #         torch.nn.functional.mse_loss(self.critic2(share_obs, actions, agent_ids_expanded), q_targets)
    #     )
    #     critic_loss = critic_loss1 + critic_loss2
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()

    #     # Add gradient clipping here (for the new model)
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
    #     torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

    #     self.critic_optimizer.step()




    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the critic."""
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)

        agent_ids = torch.arange(self.num_agents).to(share_obs.device)
        agent_ids_expanded = agent_ids.unsqueeze(0).expand(share_obs.shape[0], -1)  # (batch_size, num_agents)

        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)

        # Check for NaNs
        def has_nan(tensor):
            return torch.isnan(tensor).any().item()
        if has_nan(share_obs) or has_nan(actions) or has_nan(next_share_obs) or has_nan(next_actions):
            print("NaN detected in input tensors")
            return

        #print('target_critic')
        next_q_values1 = self.target_critic(next_share_obs, next_actions, agent_ids_expanded)
        #print(f'next_q_values1: {next_q_values1}')
        #print('target_critic2')
        next_q_values2 = self.target_critic2(next_share_obs, next_actions, agent_ids_expanded)
        #print(f'next_q_values2: {next_q_values2}')

        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)

        #print('critic')
        current_q_values1 = self.critic(share_obs, actions, agent_ids_expanded)
        #print(f'current_q_values1: {current_q_values1}')
        #print('critic2')
        current_q_values2 = self.critic2(share_obs, actions, agent_ids_expanded)
        #print(f'current_q_values2: {current_q_values2}')

        critic_loss1 = torch.mean(
            torch.nn.functional.mse_loss(current_q_values1, q_targets)
        ) + 1e-4 * sum(p.pow(2.0).sum() for p in self.critic.parameters())
        critic_loss2 = torch.mean(
            torch.nn.functional.mse_loss(current_q_values2, q_targets)
        ) + 1e-4 * sum(p.pow(2.0).sum() for p in self.critic2.parameters())
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        self.critic_optimizer.step()
        #print(f"Critic loss1: {critic_loss1.item()}, Critic loss2: {critic_loss2.item()}, Total Critic loss: {critic_loss.item()}")



    def save(self, save_dir):
        """Save the model parameters."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2" + ".pt")
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir) + "/target_critic_agent2" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model parameters."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)
        critic_state_dict2 = torch.load(str(model_dir) + "/critic_agent2" + ".pt")
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(
            str(model_dir) + "/target_critic_agent2" + ".pt"
        )
        self.target_critic2.load_state_dict(target_critic_state_dict2)

    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
