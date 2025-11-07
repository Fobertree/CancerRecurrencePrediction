import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn.models import GraphSAGE, GAT

from old_graphbuilder import build_sim_graph

class SimilarityBlock(nn.Module):
    def __init__(self, x, y, K=5) -> None:
        super().__init__()
        self.graph = build_sim_graph(x=x,y=y,k=K)

        self.graph_sage = GraphSAGE(in_channels=64, out_channels=64, num_layers=4)
        self.gat = GAT(in_channels=64, out_channels=64, num_layers=4)
        self.proj = ProjectionBlock()
        
    def forward(self):
        data = self.graph
        X = self.graph_sage(data.x, data.edge_index)
        X = self.gat(X, data.edge_index)
        X = self.proj(X)
        return X

class ProjectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
        )

        # He init
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x):
        return self.model.forward(x)
