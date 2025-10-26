import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn.models import GraphSAGE, GAT

class SimilarityBlock(nn.Module):
    def __init__(self, wsi_patches) -> None:
        super().__init__()
        self.graph_sage = GraphSAGE(in_channels=64, out_channels=64, num_layers=4)
        self.gat = GAT(in_channels=64, out_channels=64, num_layers=4)
        
    def forward(self, x):
        pass