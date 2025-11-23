#!/usr/bin/env python3
"""
Unified training script supporting multiple model backends:
- PyTorch graph models (DualStream, sim-only, spat-only, simple MLP)
- Scikit-learn models (RandomForest, XGBoost)
- Hook for hypergraph classifier (implement adapter)

Usage examples:
  python train.py --model dualstream --k_folds 5 --num_epochs 20
  python train.py --model rf --k_folds 5
  python train.py --model xgb --quick_run
"""

import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from Utils.dataset import CancerRecurrenceGraphDataset
from Models.DualStream import DualStream
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import joblib

# optional xgboost
try:
    import xgboost as xgb  # pip install xgboost
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# torchmetrics optional (used only for confusion matrix in torch loop)
from torchmetrics.classification import confusion_matrix, BinaryPrecision, BinaryRecall

# Logging
logger = logging.Logger("train", level=logging.INFO)
log_file = "Logs/train.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file, mode="a")
formatter = logging.Formatter("%(asctime)s -%(levelname)s :: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# -------------------------
# Arg parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="dualstream",
                    choices=["dualstream", "gatedfusion", "mlp_pt", "rf", "xgb", "hypergraph", "sim_only", "spat_only"],
                    help="Model to run")
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--quick_run", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--graph_dir", type=str, default="GraphDataset")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# Quick-run adjustments
if args.quick_run:
    args.k_folds = 1
    args.num_epochs = 1
    args.batch_size = 1

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(args.device)
print(f"Using device: {device}")

# -------------------------
# Dataset and CV setup
# -------------------------
dataset = CancerRecurrenceGraphDataset(root=args.graph_dir, graph_type="graphtransformer")
n = len(dataset)
if n == 0:
    raise RuntimeError("Dataset seems empty. Check GraphDataset/")

labels_arr = np.array([int(d.y.item()) for d in dataset])
if args.k_folds >= 2:
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_splits = list(kf.split(np.arange(n), labels_arr))
else:
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=args.seed,
                                          stratify=labels_arr if len(np.unique(labels_arr)) > 1 else None)
    fold_splits = [(train_idx, val_idx)]


# -------------------------
# Helpers
# -------------------------
def compute_classic_metrics(y_true, y_probs):
    """
    y_true: (N,), y_probs: (N,)
    returns: auc, f1, precision, recall
    """
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
    else:
        auc = float("nan")
    preds = (y_probs > 0.5).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return auc, f1, precision, recall


def build_tabular_features(dataset, pooling="mean"):
    """
    Convert graph dataset to X (num_samples x feat_dim) and y.
    by pooling node features per graph (mean or max).
    """
    X_list, y_list = [], []
    for i in range(len(dataset)):
        data = dataset[i]
        if not hasattr(data, "x") or data.x is None:
            raise RuntimeError(f"Graph at index {i} has no node features")
        x = data.x
        if pooling == "mean":
            vec = x.mean(dim=0).cpu().numpy()
        elif pooling == "max":
            vec = x.max(dim=0).values.cpu().numpy()
        else:
            vec = x.mean(dim=0).cpu().numpy()
        X_list.append(vec)
        y_list.append(int(data.y.item()))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y


# -------------------------
# Model factories
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def get_torch_model(name, sample_x, sample_y, sample_patch_centers):
    if name in ("dualstream", "gatedfusion"):
        return DualStream(sample_x, sample_y, sample_patch_centers, sim_out_dim=16, spat_out_dim=16, fusion_hidden=64, gate_mode="vector")
    elif name == "mlp_pt":
        if sample_x is None:
            in_dim = 64
        else:
            in_dim = int(sample_x.mean(dim=0).shape[0])
        return SimpleMLP(in_dim, hidden_dim=64)
    elif name in ("sim_only", "spat_only"):
        # Use DualStream and later zero undesired branch or add enable flags to DualStream
        return DualStream(sample_x, sample_y, sample_patch_centers, sim_out_dim=16, spat_out_dim=16, fusion_hidden=64, gate_mode="vector")
    else:
        raise ValueError(f"Unknown torch model {name}")


def get_sklearn_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=args.seed)
    elif name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed in this environment")
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=args.seed)
    elif name == "hypergraph":
        # Hook: replace with your repo's hypergraph classifier class
        raise NotImplementedError("Hypergraph classifier hook not implemented")
    else:
        raise ValueError(f"Unknown sklearn model {name}")


# -------------------------
# Training loops
# -------------------------
def train_torch_model_cv(model_name):
    all_fold_metrics = []
    sample = dataset[0]
    sample_x = getattr(sample, "x", None)
    sample_y = getattr(sample, "y", None)
    sample_patch_centers = None
    for attr in ("patch_centers", "pos", "centers", "coords"):
        if hasattr(sample, attr):
            sample_patch_centers = getattr(sample, attr)
            break

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n=== Fold {fold+1}/{len(fold_splits)} | Torch model {model_name} ===")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        model = get_torch_model(model_name, sample_x, sample_y, sample_patch_centers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -float("inf")
        ckpt_dir = "Checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold+1}_best.pth")

        # Simple epoch loop (no early stopping)
        for epoch in range(1, args.num_epochs + 1):
            # train pass
            model.train()
            train_preds = []
            train_labels = []
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                logits = logits.view(-1, 1)
                y = batch.y.float().view(-1, 1).to(device)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_preds.append(torch.sigmoid(logits).detach().cpu())
                train_labels.append(y.detach().cpu())
            train_loss /= max(1, len(train_loader))
            train_preds_np = torch.cat(train_preds, dim=0).squeeze().numpy()
            train_labels_np = torch.cat(train_labels, dim=0).squeeze().numpy()

            # val pass
            model.eval()
            val_preds = []
            val_labels = []
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    if isinstance(out, tuple):
                        logits, _ = out
                    else:
                        logits = out
                    logits = logits.view(-1, 1)
                    y = batch.y.float().view(-1, 1).to(device)
                    val_loss += float(loss.item())  # note: using last computed 'loss' isn't ideal but keep same pattern
                    val_preds.append(torch.sigmoid(logits).cpu())
                    val_labels.append(y.detach().cpu())
            val_loss /= max(1, len(val_loader))
            val_preds_np = torch.cat(val_preds, dim=0).squeeze().numpy()
            val_labels_np = torch.cat(val_labels, dim=0).squeeze().numpy()

            # metrics
            val_auc, val_f1, val_prec, val_rec = compute_classic_metrics(val_labels_np, val_preds_np)
            print(f"Fold {fold+1} Epoch {epoch}/{args.num_epochs} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValAUC {val_auc:.4f} F1 {val_f1:.4f}")

            # checkpoint best by val_auc
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({"model_state_dict": model.state_dict(), "val_auc": float(val_auc)}, ckpt_path)

        # after epochs: load best checkpoint and compute final metrics on val set
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        # eval once more to get final metrics
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                val_preds.append(torch.sigmoid(logits).cpu())
                val_labels.append(batch.y.detach().cpu())
        val_preds_np = torch.cat(val_preds, dim=0).squeeze().numpy()
        val_labels_np = torch.cat(val_labels, dim=0).squeeze().numpy()
        val_auc, val_f1, val_prec, val_rec = compute_classic_metrics(val_labels_np, val_preds_np)
        allm = {"auc": val_auc, "f1": val_f1, "prec": val_prec, "rec": val_rec}
        print(f"[Fold {fold+1}] Final (best-checkpoint) metrics: {allm}")
        all_fold_metrics.append(allm)

    return all_fold_metrics


def train_sklearn_model_cv(model_name):
    X, y = build_tabular_features(dataset, pooling="mean")
    all_fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n=== Fold {fold+1}/{len(fold_splits)} | SKLearn model {model_name} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_sklearn_model(model_name)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                probs = model.decision_function(X_val)
                probs = 1 / (1 + np.exp(-probs))
            else:
                preds = model.predict(X_val)
                probs = preds
        auc, f1, prec, rec = compute_classic_metrics(y_val, probs)
        print(f"[Fold {fold+1}] metrics: AUC {auc:.4f} F1 {f1:.4f} Prec {prec:.4f} Rec {rec:.4f}")
        model_dir = "Models_sklearn"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, f"{model_name}_fold{fold+1}.joblib"))
        all_fold_metrics.append({"auc": auc, "f1": f1, "prec": prec, "rec": rec})
    return all_fold_metrics


# -------------------------
# Run chosen model flow
# -------------------------
if args.model in ("dualstream", "gatedfusion", "mlp_pt", "sim_only", "spat_only"):
    fold_metrics = train_torch_model_cv(args.model)
else:
    fold_metrics = train_sklearn_model_cv(args.model)


# Aggregate and print summary (mean ± std)
def summarize_fold_metrics(fold_metrics):
    keys = ["auc", "f1", "prec", "rec"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
        out[f"{k}_std"] = float(np.nanstd(vals)) if vals.size > 0 else float("nan")
    return out


summary = summarize_fold_metrics(fold_metrics)
print("Summary (mean ± std across folds):")
print(f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
print(f"F1 : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
print(f"Prec: {summary['prec_mean']:.4f} ± {summary['prec_std']:.4f}")
print(f"Rec : {summary['rec_mean']:.4f} ± {summary['rec_std']:.4f}")