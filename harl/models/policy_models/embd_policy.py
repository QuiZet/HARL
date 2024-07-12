import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, embedding_dim, num_heads, num_agents):
        super(FeatureExtractor, self).__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )
        self.embedding_layer = nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
        self.combined_dim = obs_dim + embedding_dim
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.combined_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.combined_dim)
        )
        
    def forward(self, O_i, agent_ids):
        batch_size, num_agents, obs_dim = O_i.shape
        # Process input
        O_i = self.mlp_in(O_i)
        # Repeat agent IDs for the batch
        agent_ids_expanded = agent_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_agents)        
        #agent_embeddings = self.embedding_layer(agent_ids)  # (batch_size, num_agents, embedding_dim)
        agent_embeddings = self.embedding_layer(agent_ids_expanded)  # (batch_size, num_agents, embedding_dim)
        #agent_embeddings = agent_embeddings.expand(batch_size, num_agents, -1)  # (batch_size, num_agents, embedding_dim)
        O_i = torch.cat((O_i, agent_embeddings), dim=2).permute(1, 0, 2)  # (num_agents, batch_size, combined_dim)
        attn_output, _ = self.attention_layer(O_i, O_i, O_i)
        X_i = self.mlp(attn_output.permute(1, 0, 2))
        X_i = torch.mean(X_i, dim=1)  # Aggregate across num_agents
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
        # share from an existing model
        if 'model' in args:
            print('EmdbPolicyNetwork shared')
            model = args['model']
            self.feature_extractor = model.feature_extractor
            self.ensemble_policy_heads = model.ensemble_policy_heads
            self.embedding_layer = nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
            self.feature_extractor.embedding_layer = self.embedding_layer
        else:
            print('EmdbPolicyNetwork new')
            self.feature_extractor = FeatureExtractor(obs_dim=input_dim, embedding_dim=embedding_dim, num_heads=num_heads, num_agents=num_agents)
            self.ensemble_policy_heads = EnsemblePolicyHeads(input_dim=input_dim + embedding_dim, output_dim=output_dim, num_policies=num_policies)
        self.to(device)

    def forward(self, O_i, agent_ids):
        z_i = self.feature_extractor(O_i, agent_ids)  # (batch_size, num_agents, combined_dim)
        output = self.ensemble_policy_heads(z_i)
        return output
