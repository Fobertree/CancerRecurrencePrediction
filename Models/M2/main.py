# from wsi import load_images, full_patch_wsi
from .GraphTransformer import GPS

import torch
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool, global_add_pool, global_max_pool, GCNConv, SAGPooling
from torchmetrics.classification import AUROC

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from Models.M2.GatedFusion import GatedFusion
import os
import joblib

'''
GNN/GAT/GCN for local neighbors
- Then pool so we can focus on farther neighbors
    - Three candidates I want to test for pooling
        - SAGPool
        - DiffPool
        - MLAP
GraphTransformer
GNN
GraphTransformer

Test hypergraph model on metadata vs logistic regression

Or merge via a hypergraph?
'''


class ImageBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, ratio=0.5, pe_dim = 50, num_layers = 2, return_graph = False):
        super().__init__()
        #local message-passing
        self.conv1 = GATConv(pe_dim + 1, hidden_channels, heads=num_heads, concat=True)
        self.pool1 = SAGPooling(hidden_channels * num_heads, ratio=ratio)

        self.conv2 = GATConv(hidden_channels*num_heads, hidden_channels,
                             heads=num_heads, concat=True)
        self.pool2 = SAGPooling(hidden_channels * num_heads, ratio=ratio)

        #long-range
        self.gps = GPS(in_dim = in_channels, channels=in_channels + pe_dim, pe_dim=pe_dim, num_layers=num_layers,
                    attn_type='multihead', attn_kwargs={}, return_repr=True) #return_repr = True to skip MLP projection

        # self.gps_proj = nn.Linear(in_channels + pe_dim, hidden_channels * num_heads)

        # print(in_channels + pe_dim, hidden_channels * num_heads)
        self.return_graph = return_graph
        self.dropout = nn.Dropout(0.4)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # print(batch.x.shape, batch.edge_index.shape, batch.pe.shape, batch.edge_attr.shape)

        # not for node-level embeddings since DINO captured it
        # TODO: this isn't working: figure out whether it makes sense to use GPS as encoder for GAT stack
        # meant for global context
        x2 = self.dropout(self.gps(batch))
        # x2 = self.gps_proj(x2)
        # print(x2.shape, x.shape, edge_index.shape, pe.shape)
        x = torch.cat([x, batch.pe], dim=1).float()
        # print(x.shape)
        x = self.dropout(self.conv1(x, edge_index))
        batch_vec = batch.batch
        x, edge_index, edge_attr, batch_vec, _, _ = self.pool1(x, edge_index, None, batch_vec)
        x = torch.relu(x)

        x = self.dropout(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch_vec, _, _ = self.pool2(x, edge_index, None, batch_vec)
        x = torch.relu(x)

        # concat for simplicity
        # print(x.shape, x2.shape)

        # convert node level -> graph level
        x_graph = global_mean_pool(x, batch_vec)

        x = torch.cat([x_graph, x2], dim=-1)
        # print(x_graph.shape, x2.shape, x.shape)
        
        if not self.return_graph:
            # [8, 140]
            return x # [batch size, num_heads * hidden_channels + (pe_dim + in_channels)]
        
        return x, edge_index, edge_attr, batch_vec  # return pooled node features and indices
    
class MetadataBlock(nn.Module):
    def __init__(self, model_type = None, in_channels = 70):
        super().__init__()
        self.model_type = model_type
        self.model = None

        self.sklearn_clfs = ["xgb", "rf"]

        if model_type == "xgb":
            self.model = joblib.load("Models/SavedModels/xgb.pkl")
        elif model_type == "hypergraph":
            pass
        elif model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(71, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Linear(40, 20),
                nn.BatchNorm1d(20),
                nn.ReLU()
            )
            pass
        else:
            raise ValueError("invalid model type", model_type)
        # TODO: flexible init based on case/switch on model type
    
    def forward(self, X:torch.tensor):
        if self.model_type in self.sklearn_clfs:
            X_np = X.detach().cpu().numpy()
            preds = self.model.predict(X)
            preds_tensor = torch.tensor(preds, dtype=torch.float32)
            if preds.ndim == 1:
                out = preds_tensor.unsqueeze(1) # [B, 1]
        
        else:
            out = self.model(X)
            
        return out
        
# Histologic Extremely Lossy Predictor
class HELP(nn.Module):
    def __init__(self,
                 graph_in_dim,
                 hidden_channels, # for image graph
                 pe_dim,
                 metadata_dim = 70,
                 metadata_hidden_dim = 50, # doesn't apply for clf
                 num_heads=4,
                 ratio=0.8,
                 num_layers = 2):
        super().__init__()
        self.image_block = ImageBlock(graph_in_dim, hidden_channels, num_heads, ratio, num_layers=num_layers)
        img_block_dim = (graph_in_dim + pe_dim) + (num_heads * hidden_channels)

        # If you want to project metadata down to metadata_hidden_dim:
        self.metadata_block = MetadataBlock(model_type="mlp")

        self.gated_fusion = GatedFusion(img_block_dim, 20, hidden_dim=10)
        self.mlp = nn.Sequential(
            nn.Linear(10, 5),
            nn.BatchNorm1d(5), # BN to address vanishing gradients
            nn.ReLU(),
            nn.Linear(5,1)
        )

        # He init for ReLU Linear
        self.apply(kaiming_init_weights)

    def forward(self, graph_batch):
        # Run through GNN and GPS to get a graph representation
        a = self.image_block(graph_batch)
        b = self.metadata_block(graph_batch.metadata)

        # print(a.shape, b.shape)
        z, _ = self.gated_fusion(a,b)
        # print(z.shape)
        return self.mlp(z)  # logits of shape [batch_size, 1]
    
# GPS Graph Transformer: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html

def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)