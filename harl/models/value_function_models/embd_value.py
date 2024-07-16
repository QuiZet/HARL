import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from harl.utils.envs_tools import get_shape_from_obs_space


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


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
    def __init__(self, obs_dim, mlp_hidden_dim, embedding_dim, num_agents, attention_heads):
        super(FeatureExtractor, self).__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads

        self.initialize_layers(obs_dim, num_agents)

    def initialize_layers(self, obs_dim, num_agents, freeze_existing=False):
        self.obs_dim = obs_dim
        self.num_agents = num_agents

        if hasattr(self, 'mlp') and freeze_existing:
            new_mlp = MLP(obs_dim, self.mlp_hidden_dim, self.mlp_hidden_dim)
            self.mlp = self.copy_weights(self.mlp, new_mlp)
        else:
            self.mlp = MLP(obs_dim, self.mlp_hidden_dim, self.mlp_hidden_dim)

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
        agent_ids_expanded = agent_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_agents)        
        e = self.embedding(agent_ids_expanded)  # (batch_size, num_agents, embedding_dim)

        x = torch.cat((x, e), dim=-1)

        x = x.permute(1, 0, 2)
        x = self.attention(x)
        x = x.permute(1, 0, 2)

        x = self.fc_out(x)
        #x = torch.mean(x, dim=1)

        return x

class EnsemblePolicyHeads(nn.Module):
    def __init__(self, input_dim, output_dim, num_policies):
        super(EnsemblePolicyHeads, self).__init__()
        self.num_policies = num_policies
        
        # Define K shared MLPs
        self.policy_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ) for _ in range(num_policies)])
        
        # Define the attention layer to compute beta_i
        self.attention_layer = nn.Linear(input_dim, num_policies)
        
    def forward(self, z_i):
        batch_size, num_agents, combined_dim = z_i.shape
        z_i = torch.flatten(z_i, start_dim=1)  # (batch_size, num_agents * combined_dim)
        
        attention_weights = F.softmax(self.attention_layer(z_i), dim=1)
        
        policy_outputs = torch.stack([policy_block(z_i) for policy_block in self.policy_blocks], dim=1)  # (batch_size, num_policies, output_dim)
        weighted_policy_output = torch.sum(attention_weights.unsqueeze(-1) * policy_outputs, dim=1)  # (batch_size, output_dim)

        return weighted_policy_output


class EmbdValueNetwork(nn.Module):
    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(EmbdValueNetwork, self).__init__()
        num_policies = args["num_policies"]
        num_heads = args["num_heads"]
        self.num_agents = args["num_agents"]
        embedding_dim = args["embedding_dim"]
        output_dim = args["output_dim"]
        obs_dim_resized = args["obs_dim_resized"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        comb_dims = get_combined_dim(cent_obs_shape[0], act_spaces)

        if 'model' in args:
            print('EmdbValueNetwork shared')
            model = args['model']
            self.feature_extractor = model.feature_extractor
            self.ensemble_policy_heads = model.ensemble_policy_heads
            #self.embedding_layer = nn.Embedding(num_embeddings=self.num_agents, embedding_dim=embedding_dim)
            #self.feature_extractor.embedding_layer = self.embedding_layer
        else:
            print('EmdbValueNetwork new')
            print(f'comb_dims:{comb_dims} self.num_agents:{self.num_agents} num_heads:{num_heads}')
            self.feature_extractor = FeatureExtractor(obs_dim=int(comb_dims / self.num_agents), 
                                                      mlp_hidden_dim=obs_dim_resized,
                                                    embedding_dim=embedding_dim, 
                                                    attention_heads=num_heads, 
                                                    num_agents=self.num_agents)
            self.ensemble_policy_heads = EnsemblePolicyHeads(input_dim=obs_dim_resized * self.num_agents, output_dim=output_dim, num_policies=num_policies)
        self.to(device)

    def forward(self, cent_obs, actions, agent_ids):
        # Step 1: Rearrange observations to [batch, num_agents, dim]
        observations_rearranged = rearrange(cent_obs, 'b (c d) -> b c d', c=self.num_agents)  # Shape: [batch, num_agents, dim]

        # Step 2: Expand actions to match the second dimension of observations_rearranged
        actions_expanded = repeat(actions, 'b (c d) -> b c d', c=self.num_agents)  # Shape: [batch, num_agents, actions]

        # Step 3: Concatenate observations and actions along the last dimension using einops
        combined_tensor = torch.cat((observations_rearranged, actions_expanded), dim=-1)  # Shape: [batch, num_agents, dim+actions]

        z_i = self.feature_extractor(combined_tensor, agent_ids)  # (batch_size, num_agents, combined_dim)
        output = self.ensemble_policy_heads(z_i)
        return output