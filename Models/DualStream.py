import torch
import torch.nn as nn

from SimilarityBlock import SimilarityBlock
from SpatialBlock import SpatialBlock

class DualStream(nn.Module):
    def __init__(self, x, y, patch_centers, spat_K=4, sim_K=5):
        self.sim_block = SimilarityBlock(x,y, sim_K)
        self.spatial_block = SpatialBlock(x,patch_centers, y, spat_K)
        self.mlp = MLP()

    def forward(self):
        sim_out = self.sim_block()
        spat_out = self.spatial_block()

        # combine outputs, either nn or concat
        # concat for now - vstack from two 16 channel outputs
        out = torch.stack([sim_out, spat_out], dim=0) # [2, 16]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,8),
        )

        # He init
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x):
        return self.model.forward(x)