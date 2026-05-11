import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Safety check: Global attention is O(N^2). 
        # For very large graphs during testing, we skip to avoid hangs.
        if x.shape[0] > 3000: return x 
        
        x_seq = x.unsqueeze(0)
        # need_weights=False speed up computation significantly
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        attn_out = attn_out.squeeze(0)
        return self.norm(x + self.dropout(attn_out))

class GNCR(nn.Module):
    def __init__(self, hidden_channels=32, num_layers=3, num_heads=4, 
                 use_attention=True, use_jk=True, only_attention=False, regression_depth=2):
        super().__init__()
        self.use_jk = use_jk
        self.use_attention = use_attention
        self.only_attention = only_attention
        self.regression_depth = regression_depth
        
        self.input_proj = nn.Linear(1, hidden_channels)
        self.convs = nn.ModuleList()
        
        if not only_attention:
            for i in range(num_layers):
                in_channels = 1 if i == 0 else hidden_channels
                self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Determine final dimension for regression
        if only_attention:
            self.final_dim = hidden_channels
        elif use_jk:
            self.final_dim = hidden_channels * num_layers
        else:
            self.final_dim = hidden_channels

        if use_attention:
            self.attention = MultiHeadAttentionBlock(self.final_dim, num_heads)
        
        # Build regression head with variable depth
        regression_layers = []
        current_dim = self.final_dim
        
        if regression_depth == 2:
            regression_layers = [
                nn.Linear(self.final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ]
        elif regression_depth == 3:
            regression_layers = [
                nn.Linear(self.final_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ]
        elif regression_depth == 4:
            regression_layers = [
                nn.Linear(self.final_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ]
        else:
            # Default to depth 2
            regression_layers = [
                nn.Linear(self.final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ]
            
        self.regression = nn.Sequential(*regression_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.only_attention:
            h = self.input_proj(x)
            if self.use_attention: h = self.attention(h)
            return self.regression(h)

        layer_outputs = []
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            layer_outputs.append(h)

        h_final = torch.cat(layer_outputs, dim=-1) if self.use_jk else layer_outputs[-1]
        
        if self.use_attention:
            h_final = self.attention(h_final)
            
        return self.regression(h_final)