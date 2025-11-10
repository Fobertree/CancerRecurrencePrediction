import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from Utils.dataset import CancerRecurrenceGraphDataset

class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.5):
        super(SimpleGAT, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()