from wsi import load_images, full_patch_wsi
from GraphTransformer import GPS

import torch
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool
from torchmetrics.classification import AUROC

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import os

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
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.pool1 = TopKPooling(hidden_channels * num_heads, ratio=ratio)

        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels,
                             heads=num_heads, concat=True)
        self.pool2 = TopKPooling(hidden_channels * num_heads, ratio=ratio)

    def forward(self, x, edge_index, batch, edge_attr=None, pe=None):
        x = self.conv1(x, edge_index)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = torch.relu(x)

        return x, edge_index, edge_attr, batch  # return pooled node features and indices


class TestModelWithMetadata(nn.Module):
    def __init__(self,
                 graph_in_dim,
                 metadata_dim,
                 graph_hidden_dim,
                 metadata_hidden_dim,
                 hidden_channels=128,
                 hidden_dim=64,
                 num_heads=4,
                 ratio=0.8):
        super().__init__()
        # GNN backbone and GPS encoder (return_repr=True)
        self.gnn = TestImageModel(graph_in_dim, hidden_channels, num_heads, ratio)
        self.gps = GPS(channels=graph_hidden_dim, pe_dim=0, num_layers=3,
                       attn_type='performer', attn_kwargs={}, return_repr=True)

        # If you want to project metadata down to metadata_hidden_dim:
        self.meta_proj = nn.Linear(metadata_dim, metadata_hidden_dim)

        self.lin = nn.Linear(graph_hidden_dim + metadata_hidden_dim)

        # Single linear layer for logistic regression on concatenated features
        self.classifier = nn.Linear(graph_hidden_dim + metadata_hidden_dim, 1)

    def forward(self, data):
        # Run through GNN and GPS to get a graph representation
        x, edge_index, edge_attr, batch = self.gnn(data.x, data.edge_index, data.batch, data.edge_attr)
        graph_repr = self.gps(x, pe=None, edge_index=edge_index,
                              edge_attr=edge_attr, batch=batch)  # shape [batch_size, graph_hidden_dim]

        # Project metadata (or use data.metadata directly if you skip projection)
        meta_repr = self.meta_proj(data.metadata)  # shape [batch_size, metadata_hidden_dim]

        # Concatenate and apply logistic regression
        z = torch.cat((graph_repr, meta_repr), dim=-1)
        z = self.lin(z)
        z = self.lin(z)
        return self.classifier(z)  # logits of shape [batch_size, 1]

# GPS Graph Transformer: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: load w/ metadata, add confusion matrix metrics
    slides, metadata, labels = zip(*load_images("Data"))
    dataset = SlideDataset(slides, metadata, labels)

    # convert the dataset’s labels to a tensor of floats (0/1)
    all_labels = torch.stack(dataset.labels).float()

    # count positives and negatives
    pos_count = all_labels.sum().item()
    neg_count = len(all_labels) - pos_count

    # compute pos_weight = n_neg / n_pos for BCEWithLogitsLoss
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    # determine dimensionalities
    metadata_dim = len(metadata[0])
    graph_in_dim = dataset.slides[0].x.size(1)  # assuming each Data.x has shape [num_nodes, feature_dim]

    #tunable hyperparameters
    hidden_channels = 128     # number of hidden channels in GAT/GNN layers
    num_heads = 4             # attention heads in the GPS layer
    ratio = 0.8               # TopKPooling ratio (keep 80 % of nodes)
    graph_hidden_dim = hidden_channels * num_heads
    metadata_hidden_dim = metadata_dim  # size of the projected metadata representation

    # build model and move to device
    model = TestModelWithMetadata(
        metadata_dim=metadata_dim,
        metadata_hidden_dim=metadata_hidden_dim,
        hidden_channels=hidden_channels,
        ratio=ratio,
        graph_in_dim=graph_in_dim,
        graph_hidden_dim=graph_hidden_dim,
        num_heads=num_heads
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                min_lr=0.00001)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    #new training loop with weighted binary cross entropy loss 
    def train(train_loader):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for data in train_loader:
            # move batch to GPU/CPU
            data = data.to(device)

            optimizer.zero_grad()
            # forward pass: model returns logits of shape [batch_size, 1]
            logits = model(data) #preds
            # ensure labels have shape [batch_size, 1] (column vector) and type float
            y = data.y.view(-1, 1).float().to(device) #labels
            # compute weighted BCE loss
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            all_preds.append(logits)
            all_labels.append(y)

            running_loss += loss.item() * data.num_graphs
        
        # flatten
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        auroc = AUROC(task="binary")
        auc_score = auroc(all_preds, all_labels)

        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return running_loss / len(train_loader.dataset), auc_score, acc, f1

    @torch.no_grad()
    def test(loader, device):
        model.eval() # freeze train
        all_preds, all_labels = [], []

        for data in loader:
            data = data.to(device)
            logits = model(data)
            preds = torch.sigmoid(logits).round().cpu().numpy()
            # not sure if there will be a shape mismatch here
            # and if we need to do view(-1,1) for col. vec
            labels = data.y.cpu().numpy() 
            all_preds.append(preds)
            all_labels.append(labels)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        auroc = AUROC(task="binary")
        auc_score = auroc(all_preds, all_labels)

        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        return cm, auc_score, acc, f1

    # holdout
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=42)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    train_indices = np.arange(len(dataset_train))

    for i, (train_index, valid_index) in enumerate(kf.split(train_indices)):
        train_set = Subset(dataset_train, train_index)
        valid_set = Subset(dataset_train, valid_index)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
    
    os.mkdir('Weights', exists=True)

    for epoch in range(1, 101):
        # train handles full training loop
        loss, auc_score, acc, f1 = train(train_loader)
        
        # save model
        torch.save(model.state_dict(), os.path.join("Weights", f"M2Model_{epoch}.pth"))

        cm, auroc, acc, f1 = train(train_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUROC: {auroc:.4f}, F1-Score: {f1:.4f}')

    # final test
    cm, auroc, acc, f1 = test(val_loader, device)
    print(f'Test:: accuracy: {acc:.4f}, AUROC: {auroc:.4f} F1-Score: {f1:.4f}')


    pass