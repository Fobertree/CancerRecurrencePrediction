# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py

import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    Dropout
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
import torch.nn as nn


class GPS(torch.nn.Module):
    def __init__(self, in_dim, channels: int, pe_dim: int = 0, num_layers: int = 4,
                 attn_type: str = 'multihead', attn_kwargs=None, return_repr: bool = False, dropout=0.5):
        super().__init__()

        self.return_repr = return_repr
        self.use_pe = pe_dim > 0

        # Replace embeddings with linear projections
        self.node_lin = Linear(in_dim, channels - pe_dim if self.use_pe else channels)
        if self.use_pe:
            self.pe_lin = Linear(pe_dim, pe_dim)
            self.pe_norm = BatchNorm1d(pe_dim)
            self.node_emb = Embedding(2512, channels - pe_dim)

        # Optional edge projection
        self.edge_lin = Linear(1, channels)  # adjust edge feature dim if known

        # @Thomas, Akhil should I replace this basic MLP with a residual block?
        self.convs = ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(channels, channels),
                nn.Dropout(dropout),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(mlp), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            Dropout(dropout),
            ReLU(),
            Linear(channels // 2, channels // 4),
            Dropout(dropout),
            ReLU(),
            Linear(channels // 4, 1),
            nn.Sigmoid()
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # ig bc this redraw interval is so high we never redraw
        # I assume due to small # of epochs we shouldn't redraw at all?
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        pe = getattr(data, 'pe', None)

        # Only apply PE if it exists
        if pe is not None and hasattr(self, 'pe_norm') and hasattr(self, 'pe_lin'):
            x_pe = self.pe_norm(pe.float())
            x = torch.cat([self.node_emb(x.squeeze(-1)).float(), self.pe_lin(x_pe)], dim=1)
        else:
            # No PE, just embed or project nodes
            x = self.node_emb(x.squeeze(-1)) if hasattr(self, 'node_emb') else self.node_lin(x)

        if edge_attr is not None and hasattr(self, 'edge_emb'):
            edge_attr = self.edge_emb(edge_attr)
        elif edge_attr is not None and hasattr(self, 'edge_lin'):
            # hacky fix bc type mismatch float and double?
            edge_attr = self.edge_lin(edge_attr.float())

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        if self.return_repr:
            return x
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
