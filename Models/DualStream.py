import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .SimilarityBlock import SimilarityBlock
from .SpatialBlock import SpatialBlock
from .GatedFusion import GatedFusion


class DualStream(nn.Module):
    """
    DualStream model that produces similarity and spatial embeddings, fuses them
    with a gated fusion module, and passes the fused representation through an MLP.

    Forward accepts either:
      - a torch_geometric.data.Batch (recommended for training), or
      - no arg (legacy single-slide behavior where blocks were built at init).
    """
    def __init__(
        self,
        x,
        y,
        patch_centers,
        sim_out_dim: int = 16,
        spat_out_dim: int = 16,
        spat_K: int = 4,
        sim_K: int = 5,
        fusion_hidden: int = 16,
        gate_mode: str = "vector",
    ):
        super().__init__()
        # remember the expected per-node output dims from the blocks
        self.sim_out_dim = int(sim_out_dim)
        self.spat_out_dim = int(spat_out_dim)

        # instantiate the existing building blocks
        self.sim_block = SimilarityBlock(x, y, sim_K)
        self.spatial_block = SpatialBlock(x, patch_centers, y, spat_K)

        # gated fusion to combine sim and spatial embeddings
        # Note: we pass the expected input dims explicitly (GatedFusion may not expose them as attrs)
        self.fusion = GatedFusion(
            in_a_dim=self.sim_out_dim,
            in_b_dim=self.spat_out_dim,
            hidden_dim=fusion_hidden,
            gate_mode=gate_mode,
        )

        # classifier / head operating on the fused representation
        # final out_dim = 1 (single logit) for binary classification with BCEWithLogitsLoss
        self.mlp = MLP(input_dim=fusion_hidden, out_dim=1)

    def forward(self, batch=None):
        # Get node-level outputs from each block. Blocks support batch-aware forward.
        sim_nodes = self.sim_block(batch)      # expect (N_total, D_sim) or (1, D_sim)
        spat_nodes = self.spatial_block(batch) # expect (N_total, D_spat) or (1, D_spat)

        if sim_nodes is None or spat_nodes is None:
            raise RuntimeError("SimilarityBlock or SpatialBlock returned None")

        # If batch is provided and has 'batch' vector, pool per-graph
        if batch is not None and hasattr(batch, "batch"):
            batch_vec = batch.batch
            sim_pooled = global_mean_pool(sim_nodes, batch_vec)   # (B, D_sim)
            spat_pooled = global_mean_pool(spat_nodes, batch_vec) # (B, D_spat)
        else:
            # Single graph case: if node-level outputs, pool across nodes
            if sim_nodes.dim() == 2:
                sim_pooled = sim_nodes.mean(dim=0, keepdim=True)   # (1, D_sim)
            else:
                sim_pooled = sim_nodes

            if spat_nodes.dim() == 2:
                spat_pooled = spat_nodes.mean(dim=0, keepdim=True) # (1, D_spat)
            else:
                spat_pooled = spat_nodes

        # sanity checks: dims must match the expected sim_out_dim / spat_out_dim
        if sim_pooled.size(1) != self.sim_out_dim or spat_pooled.size(1) != self.spat_out_dim:
            raise RuntimeError(
                f"DualStream: pooled dims mismatch. "
                f"Got sim_pooled dim {sim_pooled.size(1)} (expected {self.sim_out_dim}), "
                f"spat_pooled dim {spat_pooled.size(1)} (expected {self.spat_out_dim}). "
                "Adjust SimilarityBlock/SpatialBlock projection output sizes or supply matching sim_out_dim/spat_out_dim when constructing DualStream."
            )

        # fuse the two modality embeddings using the gated fusion module
        fused, gate = self.fusion(sim_pooled, spat_pooled)  # fused: (B, fusion_hidden)

        # classification / prediction -> single logit per graph
        out = self.mlp(fused)  # (B, 1)

        # return logits (no sigmoid) and gate for monitoring
        return out, {"gate": gate}


class MLP(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 16, out_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # He init for linear layers
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)