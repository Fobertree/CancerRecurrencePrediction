import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

# Use the canonical graphbuilder (replace legacy import)
from Utils.graphbuilder import build_spatial_graph


class SpatialBlock(nn.Module):
    """
    Spatial branch: build a spatial graph from patch coordinates (if provided)
    or use a Batch/Data passed at forward time. Runs a small GNN stack and
    projects per-node outputs to a fixed dimension (out_dim, default 16).
    """

    def __init__(
        self,
        x: torch.Tensor | None,
        patch_centers,
        slide_labels=None,
        K: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 16,
        num_sage_layers: int = 2,
        num_gat_layers: int = 1,
        gat_heads: int = 1,
    ) -> None:
        super().__init__()

        # infer input feature size from provided x if possible (fallback to hidden_dim)
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

        # Optionally build a graph at init if patch_centers and x are provided.
        # Otherwise, require a batch Data in forward().
        self.graph: Data | None = None
        if patch_centers is not None:
            try:
                # build_spatial_graph expects patch_centers, x, radius/k signature
                # Accept either (patch_centers, x) or (patch_centers only) depending on helper
                self.graph = build_spatial_graph(patch_centers=patch_centers, x=x, radius=K)
            except Exception:
                # leave self.graph None so forward must receive a batch
                self.graph = None

        # Build a small SAGEConv stack
        sage_layers = []
        last_dim = in_channels
        for _ in range(num_sage_layers):
            sage_layers.append(SAGEConv(last_dim, last_dim))
        self.sage_layers = nn.ModuleList(sage_layers)

        # Build a small GAT stack
        gat_layers = []
        for _ in range(num_gat_layers):
            # Use GATConv with heads; keep output dims compatible by using last_dim//heads per-head
            out_per_head = max(1, last_dim // max(1, gat_heads))
            gat_layers.append(GATConv(last_dim, out_per_head, heads=gat_heads))
            # after GATConv with heads>1 output dim = out_per_head * heads; this will be close to last_dim
            # if shapes differ slightly, ProjectionBlock will handle it
        self.gat_layers = nn.ModuleList(gat_layers)

        # projection: map per-node features to out_dim embedding (match DualStream expectations)
        self.proj = ProjectionBlock(in_dim=last_dim, mid_dim=max(32, last_dim // 2), out_dim=out_dim)

    def forward(self, batch: Data | None = None):
        """
        If `batch` (torch_geometric.data.Data or Batch) is provided, use its x/edge_index.
        Otherwise, use the pre-built self.graph (if available).
        Returns node-level embeddings with shape (N, out_dim).
        """
        if batch is not None:
            data = batch
        else:
            if self.graph is None:
                raise RuntimeError(
                    "SpatialBlock: no graph available. Provide patch_centers/x at init or pass a batch Data to forward()."
                )
            data = self.graph

        if not hasattr(data, "x") or not hasattr(data, "edge_index"):
            raise ValueError("SpatialBlock.forward: data must have 'x' and 'edge_index' attributes")

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
    def __init__(self, in_dim: int = 64, mid_dim: int = 32, out_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        mid = mid_dim if mid_dim is not None else max(32, in_dim // 2)
        self.model = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim),
        )

        # He init for linear layers
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)