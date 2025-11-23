import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

# Use the new canonical graphbuilder
from Utils.graphbuilder import build_similarity_graph

class SimilarityBlock(nn.Module):
    """
    Similarity branch: builds a similarity graph from patch features (if provided)
    or uses the batch Data provided at forward time. Runs a small GNN stack and
    projects per-node outputs to a fixed dimension (out_dim, default 16).
    """

    def __init__(self, x=None, y=None, K: int = 5, hidden_dim: int = 64, out_dim: int = 16, num_sage_layers: int = 2, num_gat_layers: int = 1, gat_heads: int = 1):
        super().__init__()

        # infer input feature size from provided x if possible (fallback to 64)
        if x is None:
            in_channels = hidden_dim
        else:
            try:
                in_channels = int(x.shape[1])
            except Exception:
                in_channels = hidden_dim

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.K = K

        # Optionally build a graph at init if x is provided. Otherwise lazy-build at forward.
        self.graph: Data | None = None
        if x is not None:
            try:
                self.graph = build_similarity_graph(torch.as_tensor(x, dtype=torch.float32), k=self.K)
            except Exception:
                # fallback: leave graph None and build in forward
                self.graph = None

        # Build a small SAGEConv stack
        sage_layers = []
        last_dim = in_channels
        for i in range(num_sage_layers):
            sage_layers.append(SAGEConv(last_dim, last_dim))
        self.sage_layers = nn.ModuleList(sage_layers)

        # Build a small GAT stack
        gat_layers = []
        for i in range(num_gat_layers):
            gat_layers.append(GATConv(last_dim, last_dim // max(1, gat_heads), heads=gat_heads))
            # after GATConv with heads>1, output dim = out_per_head * heads; we keep shapes consistent by using last_dim // heads
        self.gat_layers = nn.ModuleList(gat_layers)

        # projection: map per-node features to out_dim embedding (match DualStream expectations)
        self.proj = ProjectionBlock(in_dim=last_dim, mid_dim=max(32, last_dim // 2), out_dim=out_dim)

    def forward(self, batch: Data | None = None):
        """
        If batch (torch_geometric.data.Data or Batch) is provided, use its x/edge_index.
        Otherwise, use the pre-built self.graph (built at init). Returns node-level embeddings (N, out_dim).
        """
        if batch is not None:
            data = batch
        else:
            if self.graph is None:
                raise RuntimeError("No graph available: either provide x at init or pass a batch Data to forward()")
            data = self.graph

        X = data.x
        edge_index = data.edge_index

        # apply SAGE convs
        for conv in self.sage_layers:
            X = conv(X, edge_index)
            X = torch.relu(X)

        # apply GAT convs
        for conv in self.gat_layers:
            X = conv(X, edge_index)
            X = torch.relu(X)

        X = self.proj(X)
        return X


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim=64, mid_dim=32, out_dim=16, dropout=0.3):
        super().__init__()
        # ensure dimensions are sensible
        mid = mid_dim if mid_dim is not None else max(32, in_dim // 2)
        self.model = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim),
        )

        # He init
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)