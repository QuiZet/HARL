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


class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, embedding_dim, num_heads, num_agents):
        super(FeatureExtractor, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
        self.combined_dim = obs_dim + embedding_dim
        #print(f'FE obs_dim:{obs_dim} num_heads:{num_heads}')
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.combined_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.combined_dim)
        )
        
    def forward(self, O_i, agent_ids):
        #print(f'FE O_i:{O_i.shape} agent_ids:{agent_ids}')
        batch_size, num_agents, obs_dim = O_i.shape
        # Repeat agent IDs for the batch
        agent_ids_expanded = agent_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_agents)        
        #agent_embeddings = self.embedding_layer(agent_ids)  # (batch_size, num_agents, embedding_dim)
        agent_embeddings = self.embedding_layer(agent_ids_expanded)  # (batch_size, num_agents, embedding_dim)
        #agent_embeddings = agent_embeddings.expand(batch_size, num_agents, -1)  # (batch_size, num_agents, embedding_dim)
        O_i = torch.cat((O_i, agent_embeddings), dim=2).permute(1, 0, 2)  # (num_agents, batch_size, combined_dim)
        attn_output, _ = self.attention_layer(O_i, O_i, O_i)
        X_i = self.mlp(attn_output.permute(1, 0, 2))
        return X_i

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
        #print(f'FE cent_obs_space t:{type(cent_obs_space)}')
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        comb_dims = get_combined_dim(cent_obs_shape[0], act_spaces)

        if 'model' in args:
            print('EmdbValueNetwork shared')
            model = args['model']
            self.feature_extractor = model.feature_extractor
            self.ensemble_policy_heads = model.ensemble_policy_heads
            self.embedding_layer = nn.Embedding(num_embeddings=self.num_agents, embedding_dim=embedding_dim)
            self.feature_extractor.embedding_layer = self.embedding_layer
        else:
            print('EmdbValueNetwork new')
            self.feature_extractor = FeatureExtractor(obs_dim=int(comb_dims / self.num_agents), 
                                                    embedding_dim=embedding_dim, 
                                                    num_heads=num_heads, 
                                                    num_agents=self.num_agents)
            self.ensemble_policy_heads = EnsemblePolicyHeads(input_dim=comb_dims + embedding_dim * self.num_agents, output_dim=output_dim, num_policies=num_policies)
        self.to(device)

    def forward(self, cent_obs, actions, agent_ids):
        #print(f'FE cent_obs_shape:{cent_obs.shape}')

        # Step 1: Rearrange observations to [2, 3, 18]
        observations_rearranged = rearrange(cent_obs, 'b (c d) -> b c d', c=self.num_agents)  # Shape: [2, 3, 18]
        #print(f"Rearranged observations shape: {observations_rearranged.shape}")

        # Step 2: Expand actions to match the second dimension of observations_rearranged
        #print(f"actions shape: {actions.shape}")
        #actions_expanded = repeat(actions, 'b a -> b c a', c=self.num_agents)  # Shape: [2, 3, 5]
        actions_expanded = repeat(actions, 'b (c d) -> b c d', c=self.num_agents)  # Shape: [2, 3, 5]
        #print(f"Expanded actions shape: {actions_expanded.shape}")

        # Step 3: Concatenate observations and actions along the last dimension using einops
        combined_tensor = torch.cat((observations_rearranged, actions_expanded), dim=-1)  # Shape: [2, 3, 23]
        #print(f"Combined tensor shape: {combined_tensor.shape}")

        # Print the reshaped tensor shape
        #print(f"Original tensor shape: {cent_obs.shape}")
        #print(f"combined_tensor tensor shape: {combined_tensor.shape}")

        z_i = self.feature_extractor(combined_tensor, agent_ids)  # (batch_size, num_agents, combined_dim)
        #print(f"z_i shape: {z_i.shape}")
        output = self.ensemble_policy_heads(z_i)
        return output