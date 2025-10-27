from wsi import load_images, full_patch_wsi
from GraphTransformer import GPS

import torch
import torch.functional as F
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

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

class TestImageModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, ratio):
        super().__init__()
        # MLAP pooling
        self.conv1 = GATConv(in_channels, hidden_channels, concat=True)
        self.pool1 = TopKPooling(hidden_channels * num_heads, ratio=ratio)
        
        # graph transformer
        # use performer attention for now for linear vs quadratic multihead time
        self.gps1 = GPS(hidden_channels * num_heads, pe_dim=0, num_layers=3, attn_type="performer")

        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.pool2 = TopKPooling(hidden_channels * num_heads, ratio=ratio)

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None):
        # GAT -> Pool
        x = self.conv1(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x = F.relu(x)

        # GAT -> Pool -> GPS
        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, None, batch)
        x = torch.relu(x)

        x = self.gps1.forward(x, pe, edge_index, edge_attr, batch)

        # Global pooling after hierarchical processing.
        # x = global_mean_pool(x, batch)
        return x # [1] via mlp at end of graph transformer

# GPS Graph Transformer: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html

class TestModelWithMetadata(nn.Module):
    def __init__(self, metadata_dim, hidden_channels, ratio,
                 graph_in_dim, hidden_dim=128, num_heads=4):
        # combines metadata with graph transformer model
        super().__init__()
        self.mlp = nn.Sequential(

        )
        self.graph_encoder = TestImageModel(graph_in_dim, hidden_channels, num_heads, ratio)
    
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # or num_classes
        )

    def forward(self, x, pe, edge_index, edge_attr, batch, metadata):
        graph_repr = self.graph_encoder(x, edge_index, batch, edge_attr, pe)  # [num_graphs, D]
        meta_repr = self.meta_mlp(metadata)                                   # [num_graphs, D]
        z = torch.cat((graph_repr, meta_repr), dim=-1)
        return self.classifier(z)

class SlideDataset(torch.utils.data.Dataset):
    def __init__(self, slides, metadata, labels):
        self.slides = slides
        self.metadata = [torch.tensor(m, dtype=torch.float32) for m in metadata]
        self.labels = [torch.tensor(y, dtype=torch.float32) for y in labels]

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        data = self.slides[idx]
        data.metadata = self.metadata[idx]  # attach as an attribute
        data.y = self.labels[idx]
        return data

if __name__ == "__main__":
    BATCH_SIZE = 16

    # TODO: load w/ metadata, add confusion matrix metrics
    slides, metadata, labels = zip(*load_images("Data"))
    dataset = SlideDataset(slides, metadata, labels)
    
    model = TestModelWithMetadata()
    # kfold into train, val, test data loaders
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                min_lr=0.00001)

    def train(train_loader, device):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            model.gps1.redraw_projection.redraw_projections()
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            # TODO: replace with weighted BCE loss
            loss = (out.squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader, device):
        model.eval() # freeze train
        all_preds, all_labels = [], []

        total_error = 0
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.metadata)
            preds = torch.sigmoid(logits).round().cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
            # total_error += (out.squeeze() - data.y).abs().sum().item()
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        return cm, acc, f1

        # return total_error / len(loader.dataset)

    # holdout
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=42)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE)

    for i, (train_index, valid_index) in enumerate(kf.split(dataset_train)):
        train_set = Subset(slides, train_index)
        valid_set = Subset(slides, valid_index)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(1, 101):
            loss = train(train_set)
            val_mae = test(val_loader)
            # test_mae = test(test_loader)
            scheduler.step(val_mae)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}')

    # final test
    final_test_mae = test(test_loader, device)
    print(f'Test MAE: {final_test_mae:.4f}')


    pass