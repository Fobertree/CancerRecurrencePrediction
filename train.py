import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from Utils.dataset import CancerRecurrenceGraphDataset
from Models.M2.GraphTransformer import GPS
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
graph_save_dir = "GraphDataset"
k_folds = 5
num_epochs = 50
batch_size = 4
learning_rate = 1e-3
weight_decay = 1e-5

# -----------------------------
# Load dataset
# -----------------------------
dataset = CancerRecurrenceGraphDataset(root=graph_save_dir, graph_type="combined")
num_graphs = len(dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# K-Fold Cross Validation Setup
# -----------------------------
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# For aggregating metrics across folds
all_train_losses, all_val_losses = [], []
all_train_aurocs, all_val_aurocs = [], []
all_train_f1s, all_val_f1s = [], []

# -----------------------------
# Helper function: run one epoch
# -----------------------------
def run_epoch(loader, model, criterion, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
        
    epoch_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
            model.redraw_projection.redraw_projections()
        
        edge_attr = batch.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(batch.edge_index.size(1), 1, device=batch.x.device)

        out = model(batch).squeeze().view(-1, 1)
        y = batch.y.float().view(-1, 1)

        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_preds.append(torch.sigmoid(out).detach().cpu())
        all_labels.append(y.cpu())

    avg_loss = epoch_loss / len(loader)
    all_preds_tensor = torch.cat(all_preds).numpy()
    all_labels_tensor = torch.cat(all_labels).numpy()

    # Handle single-class case
    if len(np.unique(all_labels_tensor)) > 1:
        auroc = roc_auc_score(all_labels_tensor, all_preds_tensor)
    else:
        auroc = np.nan
    f1 = f1_score(all_labels_tensor, all_preds_tensor > 0.5, zero_division=0)

    return avg_loss, auroc, f1

# -----------------------------
# Start K-Fold Cross Validation
# -----------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n=== Fold {fold + 1}/{k_folds} ===")
    
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    attn_kwargs = {'dropout': 0.5}
    model = GPS(
        in_dim=dataset.num_node_features,
        channels=64,
        pe_dim=0,
        num_layers=4,
        attn_type='multihead',
        attn_kwargs=attn_kwargs,
        return_repr=False
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    fold_train_losses, fold_val_losses = [], []
    fold_train_aurocs, fold_val_aurocs = [], []
    fold_train_f1s, fold_val_f1s = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auroc, train_f1 = run_epoch(train_loader, model, criterion, optimizer, train=True)
        val_loss, val_auroc, val_f1 = run_epoch(val_loader, model, criterion, train=False)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_aurocs.append(train_auroc)
        fold_val_aurocs.append(val_auroc)
        fold_train_f1s.append(train_f1)
        fold_val_f1s.append(val_f1)

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}")

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    all_train_aurocs.append(fold_train_aurocs)
    all_val_aurocs.append(fold_val_aurocs)
    all_train_f1s.append(fold_train_f1s)
    all_val_f1s.append(fold_val_f1s)

# -----------------------------
# Plot aggregated metrics
# -----------------------------
def plot_kfold_metrics(all_train, all_val, ylabel, title):
    mean_train = np.nanmean(all_train, axis=0)
    mean_val = np.nanmean(all_val, axis=0)
    std_train = np.nanstd(all_train, axis=0)
    std_val = np.nanstd(all_val, axis=0)

    plt.figure(figsize=(8,5))
    plt.plot(mean_train, label='Train')
    plt.fill_between(range(len(mean_train)), mean_train-std_train, mean_train+std_train, alpha=0.2)
    plt.plot(mean_val, label='Val')
    plt.fill_between(range(len(mean_val)), mean_val-std_val, mean_val+std_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

plot_kfold_metrics(all_train_losses, all_val_losses, "BCE Loss", "Training vs Validation Loss")
plot_kfold_metrics(all_train_aurocs, all_val_aurocs, "AUROC", "Training vs Validation AUROC")
plot_kfold_metrics(all_train_f1s, all_val_f1s, "F1 Score", "Training vs Validation F1 Score")
