import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    """
    Simple gated fusion of two modality embeddings.
    - Projects both inputs to same dim (if needed)
    - Computes a gating vector (or scalar) and outputs fused embedding:
        fused = gate * h_a + (1 - gate) * h_b
    - gate_mode: "scalar" or "vector"
    """
    def __init__(self, in_a_dim, in_b_dim, hidden_dim=None, gate_mode="vector", dropout=0.1):
        super().__init__()
        self.gate_mode = gate_mode
        if hidden_dim is None:
            hidden_dim = max(in_a_dim, in_b_dim)
        # projections to common size
        self.proj_a = nn.Linear(in_a_dim, hidden_dim) if in_a_dim != hidden_dim else nn.Identity()
        self.proj_b = nn.Linear(in_b_dim, hidden_dim) if in_b_dim != hidden_dim else nn.Identity()
        # gating: output dim = 1 for scalar, hidden_dim for vector
        gate_out = 1 if gate_mode == "scalar" else hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gate_out)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_a, h_b):
        """
        h_a: (B, in_a_dim)  - e.g., image/graph embedding
        h_b: (B, in_b_dim)  - e.g., metadata embedding
        returns fused: (B, hidden_dim), gate values (B, gate_out)
        """
        a = self.proj_a(h_a)
        b = self.proj_b(h_b)
        cat = torch.cat([a, b], dim=-1)
        g_logits = self.gate(cat)
        g = torch.sigmoid(g_logits)   # in (0,1)
        if self.gate_mode == "scalar":
            # make shape (B,1) -> (B,hidden_dim) for broadcast
            g = g.expand(-1, a.shape[-1])
        fused = g * a + (1.0 - g) * b
        fused = self.norm(fused)
        return fused, g