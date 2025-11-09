import torch
from torch.nn import Linear
from torch_geometric.nn import HypergraphConv, global_mean_pool

class HypergraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, hyperedge_index, batch):
        # Apply hypergraph convolutions
        x = self.conv1(x, hyperedge_index).relu()
        x = self.conv2(x, hyperedge_index).relu()

        # Global pooling for graph-level representation
        x = global_mean_pool(x, batch)

        # Linear classifier
        x = self.lin(x)
        return x