import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ----- Node Embedding Module using GraphSAGE -----
class GNCRNodeEmbedding(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.attention = torch.nn.MultiheadAttention(hidden_channels, num_heads=1)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h3 = F.relu(self.conv3(h2, edge_index))
        h_attn, _ = self.attention(h3.unsqueeze(1), h3.unsqueeze(1), h3.unsqueeze(1))
        h_attn = h_attn.squeeze(1)
        return torch.cat([h1, h2, h3, h_attn], dim=-1)

# ----- Regression Head for Node Score Prediction -----
class RegressionModule(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----- Full Model: GNCR (Graph Node Criticality Regressor) -----
class GNCR(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.embedding = GNCRNodeEmbedding(hidden_channels)
        self.regression = RegressionModule(hidden_channels * 4)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embedding = self.embedding(x, edge_index)
        return self.regression(embedding)
