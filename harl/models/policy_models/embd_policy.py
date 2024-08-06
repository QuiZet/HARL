import torch
import torch.nn as nn
import torch.nn.functional as F


from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AgentEmbedding(nn.Module):
    def __init__(self, num_agents, embedding_dim):
        super(AgentEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_agents, embedding_dim)
    
    def forward(self, agent_ids):
        return self.embedding(agent_ids)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class FeatureExtractor(nn.Module):
    def __init__(self, args, obs_dim, mlp_hidden_dim, embedding_dim, num_agents, attention_heads):
        super(FeatureExtractor, self).__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads


        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.final_activation_func = args["final_activation_func"]
        self.pi_sizes = [obs_dim] + list(self.hidden_sizes) + [self.mlp_hidden_dim]

        self.initialize_layers(obs_dim, num_agents)

    def initialize_layers(self, obs_dim, num_agents, freeze_existing=False):
        self.obs_dim = obs_dim
        self.num_agents = num_agents

        if hasattr(self, 'mlp') and freeze_existing:
            #new_mlp = MLP(obs_dim, self.mlp_hidden_dim, self.mlp_hidden_dim)
            new_mlp = PlainMLPNew(self.pi_sizes, self.activation_func, self.final_activation_func)
            self.mlp = self.copy_weights(self.mlp, new_mlp)
        else:
            #self.mlp = MLP(obs_dim, self.mlp_hidden_dim, self.mlp_hidden_dim)
            self.mlp = PlainMLPNew(self.pi_sizes, self.activation_func, self.final_activation_func)

        if hasattr(self, 'embedding') and freeze_existing:
            new_embedding = AgentEmbedding(num_agents, self.embedding_dim)
            self.embedding = self.copy_weights(self.embedding, new_embedding)
        else:
            self.embedding = AgentEmbedding(num_agents, self.embedding_dim)

        if not hasattr(self, 'attention'):
            self.attention = SelfAttention(self.mlp_hidden_dim + self.embedding_dim, self.attention_heads)
        
        if not hasattr(self, 'fc_out'):
            self.fc_out = nn.Linear(self.mlp_hidden_dim + self.embedding_dim, self.mlp_hidden_dim)

        if freeze_existing:
            self.freeze_parameters()

    def copy_weights(self, old_layer, new_layer):
        with torch.no_grad():
            if isinstance(old_layer, MLP):
                self.copy_mlp_weights(old_layer, new_layer)
            elif isinstance(old_layer, AgentEmbedding):
                self.copy_embedding_weights(old_layer, new_layer)
        return new_layer

    def copy_mlp_weights(self, old_layer, new_layer):
        with torch.no_grad():
            new_layer.fc1.weight[:old_layer.fc1.weight.shape[0], :old_layer.fc1.weight.shape[1]].copy_(old_layer.fc1.weight)
            new_layer.fc1.bias[:old_layer.fc1.bias.shape[0]].copy_(old_layer.fc1.bias)
            new_layer.fc2.weight[:old_layer.fc2.weight.shape[0], :old_layer.fc2.weight.shape[1]].copy_(old_layer.fc2.weight)
            new_layer.fc2.bias[:old_layer.fc2.bias.shape[0]].copy_(old_layer.fc2.bias)
            
            if new_layer.fc1.weight.shape[0] > old_layer.fc1.weight.shape[0]:
                avg_fc1_weight_rows = old_layer.fc1.weight.mean(dim=0)
                new_layer.fc1.weight[old_layer.fc1.weight.shape[0]:, :old_layer.fc1.weight.shape[1]].copy_(avg_fc1_weight_rows.unsqueeze(0).expand(new_layer.fc1.weight.shape[0] - old_layer.fc1.weight.shape[0], -1))
            if new_layer.fc1.weight.shape[1] > old_layer.fc1.weight.shape[1]:
                avg_fc1_weight_cols = old_layer.fc1.weight.mean(dim=1)
                new_layer.fc1.weight[:, old_layer.fc1.weight.shape[1]:].copy_(avg_fc1_weight_cols.unsqueeze(1).expand(-1, new_layer.fc1.weight.shape[1] - old_layer.fc1.weight.shape[1]))

            if new_layer.fc1.bias.shape[0] > old_layer.fc1.bias.shape[0]:
                avg_fc1_bias = old_layer.fc1.bias.mean()
                new_layer.fc1.bias[old_layer.fc1.bias.shape[0]:].fill_(avg_fc1_bias)

            if new_layer.fc2.weight.shape[0] > old_layer.fc2.weight.shape[0]:
                avg_fc2_weight_rows = old_layer.fc2.weight.mean(dim=0)
                new_layer.fc2.weight[old_layer.fc2.weight.shape[0]:, :old_layer.fc2.weight.shape[1]].copy_(avg_fc2_weight_rows.unsqueeze(0).expand(new_layer.fc2.weight.shape[0] - old_layer.fc2.weight.shape[0], -1))
            if new_layer.fc2.weight.shape[1] > old_layer.fc2.weight.shape[1]:
                avg_fc2_weight_cols = old_layer.fc2.weight.mean(dim=1)
                new_layer.fc2.weight[:, old_layer.fc2.weight.shape[1]:].copy_(avg_fc2_weight_cols.unsqueeze(1).expand(-1, new_layer.fc2.weight.shape[1] - old_layer.fc2.weight.shape[1]))

            if new_layer.fc2.bias.shape[0] > old_layer.fc2.bias.shape[0]:
                avg_fc2_bias = old_layer.fc2.bias.mean()
                new_layer.fc2.bias[old_layer.fc2.bias.shape[0]:].fill_(avg_fc2_bias)

    def copy_embedding_weights(self, old_layer, new_layer):
        new_layer.embedding.weight[:old_layer.embedding.weight.shape[0], :].copy_(old_layer.embedding.weight)
        if new_layer.embedding.weight.shape[0] > old_layer.embedding.weight.shape[0]:
            avg_embedding_weight = old_layer.embedding.weight.mean(dim=0)
            new_layer.embedding.weight[old_layer.embedding.weight.shape[0]:, :].copy_(avg_embedding_weight.unsqueeze(0).expand(new_layer.embedding.weight.shape[0] - old_layer.embedding.weight.shape[0], -1))

    def freeze_parameters(self):
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.fc_out.parameters():
            param.requires_grad = False

    def forward(self, obs, agent_ids):
        batch_size, num_agents, _ = obs.size()
        # Check if the current dimensions match the input dimensions
        if num_agents != self.num_agents or obs.size(-1) != self.obs_dim:
            self.initialize_layers(obs.size(-1), num_agents, freeze_existing=True)

        #RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = obs.reshape(batch_size * num_agents, -1)
        x = self.mlp(x)
        x = x.view(batch_size, num_agents, -1)

        # Repeat agent IDs for the batch
        #agent_ids_expanded = agent_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_agents)        
        agent_ids_expanded = agent_ids        
        e = self.embedding(agent_ids_expanded)  # (batch_size, num_agents, embedding_dim)

        x = torch.cat((x, e), dim=-1)

        x = x.permute(1, 0, 2)
        x = self.attention(x)
        x = x.permute(1, 0, 2)

        x = self.fc_out(x)
        x = torch.mean(x, dim=1)
        return x

class EnsemblePolicyHeads(nn.Module):
    def __init__(self, input_dim, output_dim, num_policies):
        super(EnsemblePolicyHeads, self).__init__()
        self.num_policies = num_policies
        
        # Define K shared MLPs
        self.policy_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ) for _ in range(num_policies)])
        
        # Define the attention layer to compute beta_i
        self.attention_layer = nn.Linear(input_dim, num_policies)
        
    def forward(self, z_i):
        #batch_size, num_agents, combined_dim = z_i.shape
        batch_size, combined_dim = z_i.shape
        z_i = torch.flatten(z_i, start_dim=1)  # (batch_size, num_agents * combined_dim)
        
        attention_weights = F.softmax(self.attention_layer(z_i), dim=1)
        
        policy_outputs = torch.stack([policy_block(z_i) for policy_block in self.policy_blocks], dim=1)  # (batch_size, num_policies, output_dim)
        weighted_policy_output = torch.sum(attention_weights.unsqueeze(-1) * policy_outputs, dim=1)  # (batch_size, output_dim)

        return weighted_policy_output


class EmbdPolicyNetwork(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super(EmbdPolicyNetwork, self).__init__()
        input_dim = args["input_dim"]
        num_heads = args["num_heads"]
        num_agents = args["num_agents"]
        num_policies = args["num_policies"]
        embedding_dim = args["embedding_dim"]
        output_dim = args["output_dim"]
        obs_dim_resized = args["obs_dim_resized"]
        # share from an existing model
        if 'model' in args:
            print('EmdbPolicyNetwork shared')
            model = args['model']
            self.feature_extractor = model.feature_extractor
            self.ensemble_policy_heads = model.ensemble_policy_heads
            # self.embedding_layer = nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
            # self.feature_extractor.embedding_layer = self.embedding_layer
        else:
            print('EmdbPolicyNetwork new')
            self.feature_extractor = FeatureExtractor(obs_dim=input_dim, mlp_hidden_dim=obs_dim_resized, 
                                                      embedding_dim=embedding_dim, attention_heads=num_heads, num_agents=num_agents)
            self.ensemble_policy_heads = EnsemblePolicyHeads(input_dim=obs_dim_resized, 
                                                             output_dim=output_dim, num_policies=num_policies)
        self.to(device)

    def forward(self, O_i, agent_ids):
        z_i = self.feature_extractor(O_i, agent_ids)  # (batch_size, num_agents, combined_dim)
        output = self.ensemble_policy_heads(z_i)
        return output





# class EmbdPolicyNetwork(nn.Module):
#     """Deterministic policy network for continuous action space."""

#     def __init__(self, args, device=torch.device("cpu")):
#         """Initialize DeterministicPolicy model.
#         Args:
#             args: (dict) arguments containing relevant model information.
#             obs_space: (gym.Space) observation space.
#             action_space: (gym.Space) action space.
#             device: (torch.device) specifies the device to run on (cpu/gpu).
#         """
#         super().__init__()

#         obs_space = args["obs_space"]
#         action_space = args["action_space"]

#         self.tpdv = dict(dtype=torch.float32, device=device)
#         hidden_sizes = args["hidden_sizes"]
#         activation_func = args["activation_func"]
#         final_activation_func = args["final_activation_func"]
#         obs_shape = get_shape_from_obs_space(obs_space)
#         if len(obs_shape) == 3:
#             self.feature_extractor = PlainCNN(
#                 obs_shape, hidden_sizes[0], activation_func
#             )
#             feature_dim = hidden_sizes[0]
#         else:
#             self.feature_extractor = None
#             feature_dim = obs_shape[0]
#         act_dim = action_space.shape[0]
#         pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
#         self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
#         low = torch.tensor(action_space.low).to(**self.tpdv)
#         high = torch.tensor(action_space.high).to(**self.tpdv)
#         self.scale = (high - low) / 2
#         self.mean = (high + low) / 2
#         self.to(device)

#     def forward(self, obs, agent_ids):
#         # Return output from network scaled to action space limits.
#         if self.feature_extractor is not None:
#             x = self.feature_extractor(obs)
#         else:
#             x = obs
#         x = self.pi(x)
#         x = self.scale * x + self.mean
#         return x


# class EnsemblePolicyHeads(nn.Module):
#     def __init__(self, args, input_dim, output_dim, num_policies):
#         super(EnsemblePolicyHeads, self).__init__()
#         self.num_policies = num_policies
        
#         hidden_sizes = args["hidden_sizes"]
#         activation_func = args["activation_func"]
#         final_activation_func = args["final_activation_func"]
#         pi_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
#         print(f'pi_size:{pi_sizes}')

#         # Define K shared MLPs
#         self.policy_blocks = nn.ModuleList([PlainMLP(pi_sizes, activation_func, final_activation_func) for _ in range(num_policies)])
        
#     def forward(self, z_i):
#         policy_outputs = torch.stack([policy_block(z_i) for policy_block in self.policy_blocks], dim=1)  # (batch_size, num_policies, output_dim)
#         averaged_policy_output = torch.mean(policy_outputs, dim=1)  # (batch_size, output_dim)
#         return averaged_policy_output



class PlainMLPNew(nn.Module):
    def __init__(self, layer_sizes, activation_func, final_activation_func):
        super(PlainMLPNew, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_func())
        layers.append(final_activation_func())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class EnsemblePolicyHeads(nn.Module):
    def __init__(self, args, input_dim, output_dim, num_policies):
        super(EnsemblePolicyHeads, self).__init__()
        self.num_policies = num_policies
        
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        pi_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        print(f'pi_size:{pi_sizes}')

        # Define K shared MLPs
        self.policy_blocks = nn.ModuleList([PlainMLPNew(pi_sizes, activation_func, final_activation_func) for _ in range(num_policies)])

        # Define the attention layer to compute beta_i
        self.attention_layer = nn.Linear(input_dim, num_policies)

    # def forward(self, z_i):
    #     policy_outputs = torch.stack([policy_block(z_i) for policy_block in self.policy_blocks], dim=1)  # (batch_size, num_policies, output_dim)
    #     averaged_policy_output = torch.mean(policy_outputs, dim=1)  # (batch_size, output_dim)
    #     return averaged_policy_output

    def forward(self, z_i):
        #print(f'EnsemblePolicyHeads z_i:{z_i}')
        z_i = torch.flatten(z_i, start_dim=1)  # (batch_size, num_agents * combined_dim)
        attention_weights = F.softmax(self.attention_layer(z_i), dim=1)
        #print(f'EnsemblePolicyHeads attention_weights:{attention_weights}')
        
        policy_outputs = torch.stack([policy_block(z_i) for policy_block in self.policy_blocks], dim=1)  # (batch_size, num_policies, output_dim)
        #print(f'EnsemblePolicyHeads policy_outputs:{policy_outputs}')
        weighted_policy_output = torch.sum(attention_weights.unsqueeze(-1) * policy_outputs, dim=1)  # (batch_size, output_dim)
        #print(f'EnsemblePolicyHeads weighted_policy_output:{weighted_policy_output}')

        # Apply sigmoid activation to ensure output is between 0 and 1
        #weighted_policy_output = torch.tanh(weighted_policy_output)
        return weighted_policy_output



# https://stackoverflow.com/questions/68547474/learnable-weighted-sum-of-tensors

class EmbdPolicyNetwork(nn.Module):
    """Deterministic policy network for continuous action space."""

    def __init__(self, args, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()

        obs_space = args["obs_space"]
        action_space = args["action_space"]

        input_dim = args["input_dim"]
        num_heads = args["num_heads"]
        num_agents = args["num_agents"]
        num_policies = args["num_policies"]
        embedding_dim = args["embedding_dim"]
        output_dim = args["output_dim"]
        obs_dim_resized = args["obs_dim_resized"]
        # share from an existing model
        if 'model' in args:
            print('EmdbPolicyNetwork shared')
            model = args['model']
            self.feature_extractor = model.feature_extractor
            self.ensemble_policy_heads = model.ensemble_policy_heads
            # self.embedding_layer = nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
            # self.feature_extractor.embedding_layer = self.embedding_layer
        else:
            print('EmdbPolicyNetwork new')
            # Define the model hyperparameters
            args_new = {
                "hidden_sizes": [128, 128],
                "activation_func": nn.ReLU,
                #"final_activation_func": nn.Identity
                "final_activation_func": nn.Tanh
            }
            self.feature_extractor = FeatureExtractor(args=args_new, obs_dim=input_dim, mlp_hidden_dim=obs_dim_resized, 
                                                      embedding_dim=embedding_dim, attention_heads=num_heads, num_agents=num_agents)
            self.ensemble_policy_heads = EnsemblePolicyHeads(args=args_new,
                                                             input_dim=obs_dim_resized, 
                                                             output_dim=output_dim, num_policies=num_policies)

        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        # if len(obs_shape) == 3:
        #     self.feature_extractor = PlainCNN(
        #         obs_shape, hidden_sizes[0], activation_func
        #     )
        #     feature_dim = hidden_sizes[0]
        # else:
        #     self.feature_extractor = None
        #     feature_dim = obs_shape[0]
        # act_dim = action_space.shape[0]
        # pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]

        act_dim = action_space.shape[0]
        pi_sizes = [obs_dim_resized] + list(hidden_sizes) + [act_dim]
        print(f'pi_size:{pi_sizes}')
        self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(self, obs, agent_ids):
        #print(f'obs:{obs.shape}')
        # Return output from network scaled to action space limits.
        # if self.feature_extractor is not None:
        #     x = self.feature_extractor(obs)
        # else:
        #     x = obs
        # x = self.pi(x)
        # x = self.scale * x + self.mean

        z_i = self.feature_extractor(obs, agent_ids)  # (batch_size, num_agents, combined_dim)
        #x = self.pi(z_i)
        x = self.ensemble_policy_heads(z_i)
        #x = self.scale * x + self.mean

        #x = torch.sigmoid(x)

        # Output clipping
        #x = torch.clamp(x, min=0.0, max=1.0)
        #x = torch.clamp(x, min=-1.0, max=1.0)

        #print(f'x:{x.shape}')
        return x
    



# class EmbdPolicyNetwork(nn.Module):
#     """Deterministic policy network for continuous action space."""

#     def __init__(self, args, device=torch.device("cpu")):
#         """Initialize DeterministicPolicy model.
#         Args:
#             args: (dict) arguments containing relevant model information.
#             obs_space: (gym.Space) observation space.
#             action_space: (gym.Space) action space.
#             device: (torch.device) specifies the device to run on (cpu/gpu).
#         """
#         print('>>> Temporary EmbdPolicyNetwork')
#         super().__init__()

#         obs_space = args["obs_space"]
#         action_space = args["action_space"]

#         input_dim = args["input_dim"]
#         num_heads = args["num_heads"]
#         num_agents = args["num_agents"]
#         num_policies = args["num_policies"]
#         embedding_dim = args["embedding_dim"]
#         output_dim = args["output_dim"]
#         obs_dim_resized = args["obs_dim_resized"]

#         self.tpdv = dict(dtype=torch.float32, device=device)
#         hidden_sizes = args["hidden_sizes"]
#         activation_func = args["activation_func"]
#         final_activation_func = args["final_activation_func"]
#         obs_shape = get_shape_from_obs_space(obs_space)
#         if len(obs_shape) == 3:
#             self.feature_extractor = PlainCNN(
#                 obs_shape, hidden_sizes[0], activation_func
#             )
#             feature_dim = hidden_sizes[0]
#         else:
#             self.feature_extractor = None
#             feature_dim = obs_shape[0]
#         act_dim = action_space.shape[0]
#         pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
#         self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
#         low = torch.tensor(action_space.low).to(**self.tpdv)
#         high = torch.tensor(action_space.high).to(**self.tpdv)
#         self.scale = (high - low) / 2
#         self.mean = (high + low) / 2
#         self.to(device)

#     def forward(self, obs, agent_ids):
#         #print(f'obs:{obs.shape}')
#         # Return output from network scaled to action space limits.
#         if self.feature_extractor is not None:
#             x = self.feature_extractor(obs)
#         else:
#             x = obs
#         x = self.pi(x)
#         x = self.scale * x + self.mean
#         #print(f'x:{x.shape}')
#         return x
