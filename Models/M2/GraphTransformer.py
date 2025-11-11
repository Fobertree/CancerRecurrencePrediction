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
            x_pe = self.pe_norm(pe)
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

if __name__ == "__main__":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC-PE')
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
    test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attn_type', default='multihead',
        help="Global attention type such as 'multihead' or 'performer'.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}
    model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=args.attn_type,
                attn_kwargs=attn_kwargs, return_repr=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                min_lr=0.00001)


    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            model.redraw_projection.redraw_projections()
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            loss = (out.squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            total_error += (out.squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)


    for epoch in range(1, 101):
        loss = train()
        val_mae = test(val_loader)
        test_mae = test(test_loader)
        scheduler.step(val_mae)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
            f'Test: {test_mae:.4f}')