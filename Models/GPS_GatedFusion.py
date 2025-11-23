import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# import your GatedFusion module (expects Models/GatedFusion.py to define GatedFusion)
try:
    from Models.GatedFusion import GatedFusion
except Exception as e:
    raise ImportError("Failed to import GatedFusion from Models.GatedFusion: " + repr(e))

# import GPS (GraphTransformer) from your repo
try:
    from Models.M2.GraphTransformer import GPS
except Exception as e:
    raise ImportError("Failed to import GPS from Models.M2.GraphTransformer: " + repr(e))


class SimpleSpatialEncoder(nn.Module):
    """Fallback spatial encoder: mean-pool node features and an MLP."""
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, batch):
        if not hasattr(batch, "x") or batch.x is None:
            raise RuntimeError("Batch has no node features for SimpleSpatialEncoder")
        pooled = global_mean_pool(batch.x, batch.batch)  # (B, feat)
        return self.net(pooled)  # (B, out_dim)


class GPSGatedFusion(nn.Module):
    """
    Wrapper: GPS (sim encoder) + spatial encoder + GatedFusion + head.
    Returns (logits, info_dict) where info_dict includes 'gate' (if available).
    """
    def __init__(self, gps_cfg: dict, spat_in_dim: int,
                 sim_out_dim: int = 64, spat_out_dim: int = 64, fusion_hidden: int = 64,
                 gate_mode: str = "vector", dropout: float = 0.2, freeze_gps: bool = False):
        super().__init__()
        if gps_cfg is None:
            raise ValueError("gps_cfg must be provided for GPSGatedFusion")

        # Ensure GPS returns representations (pooled graph embedding)
        gps_cfg = gps_cfg.copy()
        gps_cfg.setdefault("return_repr", True)
        # instantiate GPS
        self.gps = GPS(**gps_cfg)

        # determine GPS output dim (best-effort)
        # many GPS implementations use 'channels' as the representation dim
        sim_dim = gps_cfg.get("channels", None)
        if sim_dim is None:
            # try introspection fallbacks
            if hasattr(self.gps, "node_lin"):
                try:
                    sim_dim = int(self.gps.node_lin.out_features)
                except Exception:
                    sim_dim = 128
            else:
                sim_dim = 128

        # optional projection to sim_out_dim
        self.sim_proj = nn.Linear(sim_dim, sim_out_dim) if sim_dim != sim_out_dim else nn.Identity()

        # spatial encoder (replace with your SpatialBlock if present)
        self.spat_enc = SimpleSpatialEncoder(in_dim=spat_in_dim, hidden_dim=spat_out_dim, out_dim=spat_out_dim, dropout=dropout)

        # gated fusion (uses your GatedFusion implementation)
        self.fusion = GatedFusion(in_a_dim=sim_out_dim, in_b_dim=spat_out_dim, hidden_dim=fusion_hidden, gate_mode=gate_mode, dropout=dropout)

        # classification head -> raw logit
        self.head = nn.Sequential(
            nn.Linear(fusion_hidden, max(8, fusion_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, fusion_hidden // 2), 1)
        )

        if freeze_gps:
            for p in self.gps.parameters():
                p.requires_grad = False

    def forward(self, batch):
        """
        Input: torch_geometric Batch
        Output: (logits, info_dict)
          - logits: (B,1) raw logits (no sigmoid)
          - info_dict: { 'gate': gate_tensor (B,1 or B,H) } or {}
        """
        # sim encoder: GPS should return pooled graph repr when return_repr=True
        sim_repr = self.gps(batch)
        # GPS implementations may return tensors or (repr, info); handle both
        if isinstance(sim_repr, tuple):
            sim_repr = sim_repr[0]

        if sim_repr is None or not torch.is_tensor(sim_repr):
            raise RuntimeError("GPS.forward did not return a tensor repr. Adapt GPSGatedFusion to extract representation.")

        sim_vec = self.sim_proj(sim_repr)  # (B, sim_out_dim)

        # spatial encoder: pooled node features -> vector
        spat_vec = self.spat_enc(batch)  # (B, spat_out_dim)

        # fusion (returns fused vector and gate values)
        fused, gate = self.fusion(sim_vec, spat_vec)  # fused: (B, fusion_hidden)

        logits = self.head(fused).view(-1, 1)  # (B,1)

        info = {"gate": gate}
        return logits, info