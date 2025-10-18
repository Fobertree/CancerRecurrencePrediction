import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.models import GraphSAGE, GCN

from graphbuilder import build_spatial_graph

class SpatialBlock(nn.Module):
    def __init__(self, x, patch_corners, slide_labels, spatial_radius = 512) -> None:
        '''
        patch_corners: corners of patches to create graph from
        slide_labels: binary labels >= 26 oncotype dx
        '''
        super().__init__()
        self.graph = build_spatial_graph(x, patch_corners=patch_corners, 
                                     slide_labels=slide_labels, 
                                     spatial_radius=spatial_radius)
        
        # simple mean aggregator for now
        self.graph_sage = GraphSAGE(in_channels=64, out_channels=64, num_layers=4)
        self.gat = GCN(in_channels=64, out_channels=64, num_layers=3)
        
    def forward(self):
        x, edge_index, y = self.graph
        self.graph_sage(self.graphX)
        pass

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
