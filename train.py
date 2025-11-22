import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from Utils.dataset import CancerRecurrenceGraphDataset
from Models.M2.GraphTransformer import GPS
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchmetrics.classification import confusion_matrix, BinaryPrecision, BinaryRecall
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from Utils.earlystopper import EarlyStopper
from Utils.focalloss import FocalLoss

logger = logging.Logger("train", level=logging.DEBUG)
log_file = 'Logs/train.log'
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s -%(levelname)s :: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------
# Hyperparameters
# -----------------------------
graph_save_dir = "GraphDatasetSeq"
k_folds = 5
num_epochs = 30
batch_size = 8 # low batch size to regularize
learning_rate = 1e-4
weight_decay = 1e-7 # L2 regularization param for Adam

# -----------------------------
# Load dataset
# -----------------------------
dataset = CancerRecurrenceGraphDataset(root=graph_save_dir, graph_type="graphtransformer")
num_graphs = len(dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# K-Fold Cross Validation Setup
# -----------------------------
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# For aggregating metrics across folds
all_train_losses, all_val_losses = [], []
all_train_aurocs, all_val_aurocs = [], []
all_train_f1s, all_val_f1s = [], []
all_train_precisions, all_val_precisions = [],[]
all_train_recalls, all_val_recalls = [],[]

early_stopper = EarlyStopper(patience=5, min_delta=5)

# -----------------------------
# Helper function: run one epoch
# -----------------------------
def run_epoch(loader, model, criterion, optimizer=None, train=True, scheduler=None):
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
        
        edge_attr = batch.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(batch.edge_index.size(1), 1, device=batch.x.device)

        # print(batch)
        out = model(batch).squeeze().view(-1, 1)
        y = batch.y.float().view(-1, 1)

        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_preds.append(torch.sigmoid(out).detach().cpu())
        all_labels.append(y.cpu())
    
    if not train:
        # print test class balance
        counts = torch.bincount(torch.cat(all_labels).flatten().int())
        print("Validation/Test Set Counts:", counts)

    avg_loss = epoch_loss / len(loader)
    all_preds_tensor = torch.cat(all_preds).numpy()
    # convert logits to labels
    all_preds_tensor = torch.tensor(all_preds_tensor > 0.5).int()
    all_labels_tensor = torch.cat(all_labels).int()

    # Handle single-class case
    if len(np.unique(all_labels_tensor)) > 1:
        auroc = roc_auc_score(all_labels_tensor, all_preds_tensor)
    else:
        auroc = np.nan
    f1 = f1_score(all_labels_tensor, all_preds_tensor > 0.5, zero_division=0)

    # just doing this quick fix bc im tired
    # all_preds_tensor = torch.from_numpy(all_preds_tensor)
    # all_labels_tensor = torch.from_numpy(all_labels_tensor)
    cm_metric = confusion_matrix.BinaryConfusionMatrix()
    cm_metric.update(all_preds_tensor, all_labels_tensor)

    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    precision_metric.update(all_preds_tensor, all_labels_tensor)
    recall_metric.update(all_preds_tensor, all_labels_tensor)

    cm = cm_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()

    return avg_loss, auroc, f1, cm, precision, recall

# -----------------------------
# Start K-Fold Cross Validation
# -----------------------------

print(f"DATASET SIZE: {len(dataset.y)}")

# TODO: multiprocess K-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, dataset.y)):
    print(f"\n=== Fold {fold + 1}/{k_folds} ===")
    
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    y = dataset.y
    class_counts = np.bincount(y[train_idx])
    class_weights = 1. / class_counts
    sample_weights = class_weights[y[train_idx]]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # --- Sampler ---
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    attn_kwargs = {'dropout': 0.5}
    model = GPS(
        in_dim=dataset.num_node_features,
        channels=100,
        pe_dim=50,
        num_layers=4,
        attn_type='performer',
        attn_kwargs=attn_kwargs,
        return_repr=False,
        dropout=0.2
    ).to(device)

    all_labels = []
    for batch in train_loader:
        all_labels.append(batch.y)
    
    y_train = torch.concat(all_labels, dim=0)

    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    # logger.info(f"Class ratio: Pos: {num_pos}, Neg: {num_neg}")
    print(f"Class ratio: Pos: {num_pos}, Neg: {num_neg}")
    weight = (num_neg / (num_pos)) # weight is hyperparam
    pos_weight = torch.tensor([weight], dtype=torch.float32).to(device)
    # specificity

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCELoss()
    # criterion = FocalLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

    fold_train_losses, fold_val_losses = [], []
    fold_train_aurocs, fold_val_aurocs = [], []
    fold_train_f1s, fold_val_f1s = [], []
    fold_train_precisions, fold_val_precisions = [],[]
    fold_train_recalls, fold_val_recalls = [],[]

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auroc, train_f1, train_cm, train_precision, train_recall = run_epoch(train_loader, model, criterion, optimizer, train=True)
        val_loss, val_auroc, val_f1, val_cm, val_precision, val_recall = run_epoch(val_loader, model, criterion, train=False)
        # logger.info(f"Confusion Matrix: {str(val_cm)}")

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_aurocs.append(train_auroc)
        fold_val_aurocs.append(val_auroc)
        fold_train_f1s.append(train_f1)
        fold_val_f1s.append(val_f1)
        fold_train_precisions.append(train_precision)
        fold_val_precisions.append(val_precision)
        fold_train_recalls.append(train_recall)
        fold_val_recalls.append(val_recall)

        # logger.debug(f"Epoch {epoch}/{num_epochs} | "
        #       f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, F1: {train_f1:.4f} | "
        #       f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        print(val_cm)
        # TN FP
        # FN TP

        # see line 218 of confusion_matrix.py

        # early stopping
        if early_stopper.early_stop(validation_loss=val_loss):
            print(f"EARLY STOP AT EPOCH: {epoch}")
            break

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    all_train_aurocs.append(fold_train_aurocs)
    all_val_aurocs.append(fold_val_aurocs)
    all_train_f1s.append(fold_train_f1s)
    all_val_f1s.append(fold_val_f1s)
    all_train_precisions.append(fold_train_precisions)
    all_val_precisions.append(fold_val_precisions)
    all_train_recalls.append(fold_train_recalls)
    all_val_recalls.append(fold_val_recalls)

    logger.info(f"Fold: {fold+1} Final metrics: AUC: {np.mean(all_val_aurocs[-1]):.4f} F1: {np.mean(all_val_f1s[-1]):.4f} Precisions: {np.mean(all_val_precisions[-1]):.4f} Recalls: {np.mean(all_val_recalls[-1]):.4f}")

# -----------------------------
# Plot aggregated metrics
# -----------------------------
def plot_kfold_metrics(all_train, all_val, ylabel, title, save = True, save_path = "Plots"):
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
    
    if (save):
        plt.savefig(os.path.join(save_path, f"{title}_{time.time()}.png"))
    else:
        plt.show()

plot_kfold_metrics(all_train_losses, all_val_losses, "BCE Loss", "Training vs Validation Loss")
plot_kfold_metrics(all_train_aurocs, all_val_aurocs, "AUROC", "Training vs Validation AUROC")
plot_kfold_metrics(all_train_f1s, all_val_f1s, "F1 Score", "Training vs Validation F1 Score")

print(f"Final metrics: AUC: {np.mean(all_val_aurocs[-1]):.4f} F1: {np.mean(all_val_f1s[-1]):.4f} Precisions: {np.mean(all_val_precisions[-1]):.4f} Recalls: {np.mean(all_val_recalls[-1]):.4f}")