import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

def build_hyperedge_index_from_batch(X_batch: torch.Tensor):
    """
    X_batch: (B, N) -- rows are samples (hyperedges), columns are features (nodes)

    Returns:
      node_index: LongTensor[K]
      hyperedge_index: LongTensor[K]
      weights: FloatTensor[K]
    """
    B, N = X_batch.shape

    # All (node, hyperedge) pairs
    node_ids = torch.arange(N).repeat(B)               # shape (B*N,)
    hyperedge_ids = torch.arange(B).repeat_interleave(N)

    # Weights = feature values
    weights = X_batch.reshape(-1)                      # (B*N,)

    return node_ids, hyperedge_ids, weights

class HGNNBackbone(nn.Module):
    """
    Hypergraph GNN backbone:
      - nodes = features (N)
      - hyperedges = samples (B)
      - uses PyG's HypergraphConv

    Produces per-sample (hyperedge) embeddings.
    """
    def __init__(
        self,
        num_features: int,
        node_dim: int = 32,
        hidden_dims=[64, 64],
        dropout=0.1,
    ):
        super().__init__()

        self.num_features = num_features

        # learnable node embeddings for each feature
        self.node_emb = nn.Parameter(torch.randn(num_features, node_dim) * 0.1)

        dims = [node_dim] + hidden_dims
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for in_c, out_c in zip(dims[:-1], dims[1:]):
            self.convs.append(HypergraphConv(in_c, out_c))
            self.norms.append(nn.LayerNorm(out_c))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X_batch):
        """
        X_batch: (B, N)
        Returns:
            E: (B, hidden_dim)  sample embeddings
            X_nodes: (N, hidden_dim) final node features
        """
        node_ids, hyperedge_ids, weights = build_hyperedge_index_from_batch(X_batch)
        edge_index = torch.stack([node_ids, hyperedge_ids], dim=0)

        # initial node features
        X_nodes = self.node_emb   # (N, node_dim)

        # apply HypergraphConv layers
        for conv, norm in zip(self.convs, self.norms):
            X_nodes = conv(X_nodes, edge_index, weights)
            X_nodes = F.gelu(X_nodes)
            X_nodes = norm(X_nodes)
            X_nodes = self.dropout(X_nodes)

        # Compute per-sample embedding = weighted sum of participating nodes
        # For hyperedge j: sum_i (X_nodes[i] * X_batch[j,i])
        E = X_batch @ X_nodes      # (B, hidden_dim)

        # Average instead of sum
        degrees = X_batch.sum(dim=1, keepdim=True)
        E = E / (degrees + 1e-12)

        return E, X_nodes

class HGNNClassifier(nn.Module):
    """
    Outputs one logit per sample (use with BCEWithLogitsLoss).
    """
    def __init__(
        self,
        num_features,
        node_dim=32,
        hidden_dims=[64, 64],
        mlp_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = HGNNBackbone(
            num_features,
            node_dim=node_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        last = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(last, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, X_batch):
        E, _ = self.backbone(X_batch)
        return self.classifier(E).squeeze(1)   # (B,)


class HGNNEncoder(nn.Module):
    """
    Outputs an n-dimensional vector per sample for fusion.
    """
    def __init__(
        self,
        num_features,
        node_dim=32,
        hidden_dims=[64, 64],
        output_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = HGNNBackbone(
            num_features,
            node_dim=node_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        last = hidden_dims[-1]
        self.proj = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(last // 2, output_dim),
        )

    def forward(self, X_batch):
        E, _ = self.backbone(X_batch)
        return self.proj(E)
