from wsi import load_images, full_patch_wsi
from GraphTransformer import GPS

import torch
import torch.functional as F
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool

'''
GNN/GAT/GCN for local neighbors
- Then pool so we can focus on farther neighbors
    - Three candidates I want to test for pooling
        - SAGPool
        - DiffPool
        - MLAP
GraphTransformer
GNN
GraphTransformer

Test hypergraph model on metadata vs logistic regression

Or merge via a hypergraph?
'''

class TestImageModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, ratio):
        super().__init__()
        # MLAP pooling
        self.conv1 = GATConv(in_channels, hidden_channels, concat=True)
        self.pool1 = TopKPooling(hidden_channels * num_heads, ratio=ratio)
        
        # graph transformer
        # use performer attention for now for linear vs quadratic multihead time
        self.gps1 = GPS(hidden_channels * num_heads, pe_dim=0, num_layers=3, attn_type="performer")

        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.pool2 = TopKPooling(hidden_channels * num_heads, ratio=ratio)

        # graph transformer
        self.gps2 = GPS(hidden_channels * num_heads, pe_dim=0, num_layers=3, attn_type="performer")

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None):
        # GAT -> Pool
        x = self.conv1(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x = F.relu(x)

        # GAT -> Pool -> GPS
        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, None, batch)
        x = torch.relu(x)

        x = self.gps1.forward(x, pe, edge_index, edge_attr, batch)

        # Global pooling after hierarchical processing
        x = global_mean_pool(x, batch)
        pass

# GPS Graph Transformer: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html

class TestModelWithMetadata(nn.Module):
    def __init__(self):
        # combines metadata with graph transformer model
        super().__init__()
        self.mlp = nn.Sequential(
            
        )
    
    def forward(self):
        pass

if __name__ == "__main__":
    slides = load_images("Data")

    # kfold into train, val, test data loaders
    pass