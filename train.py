#!/usr/bin/env python3
"""
Unified training script supporting multiple model backends:
- GraphTransformer + GatedFusion (GPSGatedFusion)
- PyTorch graph models (GraphTransformer variants, GAT if available)
- Simple PyTorch MLP baseline
- Scikit-learn models (RandomForest, XGBoost)

Notes:
- This file expects Models/GPS_GatedFusion.py (GPSGatedFusion) to exist and
  Models.M2.GraphTransformer.GPS to be available for direct GraphTransformer runs.
- The fused model returns raw logits (no sigmoid) and training uses BCEWithLogitsLoss.
- AUROC is computed on probabilities (sigmoid(logits)). F1 is reported at 0.5
  and we also compute PR-AUC and the best F1 threshold on validation.
"""

import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from Utils.dataset import CancerRecurrenceGraphDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc as _auc, confusion_matrix as sk_confusion_matrix
import joblib

# optional xgboost
try:
    import xgboost as xgb  # pip install xgboost
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Import wrapper and (optionally) GPS directly
try:
    from Models.GPS_GatedFusion import GPSGatedFusion
except Exception:
    GPSGatedFusion = None

try:
    from Models.M2.GraphTransformer import GPS
except Exception:
    GPS = None

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
parser.add_argument("--model", type=str, default="gatedfusion_graphtransform",
                    choices=["gatedfusion_graphtransform", "graphtransformer_performer", "graphtransformer_mha", "mlp_pt", "rf", "xgb", "hypergraph"],
                    help="Model to run")
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-7)
parser.add_argument("--quick_run", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--graph_dir", type=str, default="GraphDataset")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--gatedfusion_freeze_gps", action="store_true", help="Freeze GPS encoder weights initially")
parser.add_argument("--imbalance_strategy", type=str, default="sampler", choices=["sampler", "pos_weight", "none"], help="How to handle class imbalance")
parser.add_argument("--num_workers", type=int, default=0)
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
    returns: auc, f1_at_0.5, precision_at_0.5, recall_at_0.5, pr_auc, best_f1, best_thresh
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

    # PR-AUC and best-F1 threshold
    pr_auc = float("nan")
    best_f1 = 0.0
    best_thresh = 0.5
    if len(np.unique(y_true)) > 1:
        precs, recs, ths = precision_recall_curve(y_true, y_probs)
        pr_auc = _auc(recs, precs)
        # compute F1s for thresholds (note: precs/recs length = len(ths)+1)
        f1s = 2 * (precs * recs) / (precs + recs + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        # map best_idx to threshold (best_idx < len(ths) typically)
        if best_idx < len(ths):
            best_thresh = float(ths[best_idx])
        else:
            best_thresh = 0.5
        best_f1 = float(f1s[best_idx])

    return auc, f1, precision, recall, pr_auc, best_f1, best_thresh


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
    """
    Factory to return a torch model that accepts a torch_geometric Batch and returns:
      - logits tensor shaped (B,1) OR
      - (logits, info_dict) where info_dict can contain 'gate' for fusion gate values
    """
    # gatedfusion with GraphTransformer (GPS)
    if name == "gatedfusion_graphtransform":
        if GPSGatedFusion is None:
            raise RuntimeError("GPSGatedFusion not available. Add Models/GPS_GatedFusion.py")
        if sample_x is None:
            spat_in_dim = dataset.num_node_features
        else:
            try:
                spat_in_dim = int(sample_x.shape[1])
            except Exception:
                spat_in_dim = dataset.num_node_features

        gps_cfg = dict(
            in_dim=spat_in_dim,
            channels=100,
            pe_dim=0,   # set 0 because dataset does not provide positional encoding tokens
            num_layers=2,
            attn_type='performer',
            attn_kwargs={'dropout': 0.5},
            return_repr=True,
            dropout=0.2
        )
        return GPSGatedFusion(gps_cfg=gps_cfg, spat_in_dim=spat_in_dim,
                              sim_out_dim=32, spat_out_dim=32, fusion_hidden=64,
                              gate_mode="vector", dropout=0.2, freeze_gps=args.gatedfusion_freeze_gps)

    # direct GraphTransformer baseline using GPS class (if present)
    elif name == "graphtransformer_performer" or name == "graphtransformer_mha":
        if GPS is None:
            raise RuntimeError("GPS class not importable (Models.M2.GraphTransformer).")
        attn_type = 'performer' if name.endswith('performer') else 'multihead'
        # GPS in repo currently produces a Sigmoid in its mlp; ideally remove final Sigmoid in GPS.mlp.
        return GPS(in_dim=dataset.num_node_features, channels=100, pe_dim=50, num_layers=2, attn_type=attn_type, attn_kwargs={'dropout':0.5}, return_repr=False, dropout=0.2)

    elif name == "mlp_pt":
        if sample_x is None:
            in_dim = 64
        else:
            in_dim = int(sample_x.mean(dim=0).shape[0])
        return SimpleMLP(in_dim, hidden_dim=64)

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
        raise NotImplementedError("Hypergraph classifier hook not implemented")
    else:
        raise ValueError(f"Unknown sklearn model {name}")


# -------------------------
# Training helpers
# -------------------------
def safe_to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_logits_if_probs(x, eps=1e-6):
    """
    If x values are in [0,1], assume they are probabilities and convert to logits.
    Otherwise assume x are logits already.
    """
    if x.numel() == 0:
        return x
    mn = float(x.min().item())
    mx = float(x.max().item())
    if 0.0 <= mn and mx <= 1.0:
        p = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(p / (1 - p))
    return x


def run_epoch(loader, model, criterion, optimizer=None, train=True):
    """
    Runs one epoch. Model forward may return logits or (logits, info).
    This function handles outputs that are probabilities (0..1) by converting
    them to logits for loss computation, but always uses probabilities for AUROC.
    Returns per-epoch aggregates and arrays needed for diagnostics.
    """
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    all_probs = []
    all_labels = []
    gate_vals = []

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()

        out = model(batch)
        if isinstance(out, tuple):
            logits_or_probs, info = out
        else:
            logits_or_probs, info = out, {}

        # ensure tensor shape (N,1)
        logits_or_probs = logits_or_probs.view(-1, 1)

        # If the model returned probabilities in [0,1], convert to logits for loss
        logits_for_loss = to_logits_if_probs(logits_or_probs)

        y = batch.y.float().view(-1, 1).to(device)
        loss = criterion(logits_for_loss, y)
        if train:
            loss.backward()
            optimizer.step()

        epoch_loss += float(loss.item())

        # Always store 1-D arrays: flatten the tensor so single-sample batches become 1-D
        probs = torch.sigmoid(logits_for_loss).detach().cpu().view(-1)  # (N,)
        labels = y.detach().cpu().view(-1).int()                       # (N,)

        all_probs.append(probs)
        all_labels.append(labels)

        # collect gate if returned in info
        if isinstance(info, dict) and 'gate' in info and info['gate'] is not None:
            gate_vals.append(info['gate'].detach().cpu().view(info['gate'].shape[0], -1))

    avg_loss = epoch_loss / max(1, len(loader))

    if len(all_probs) == 0:
        # no data
        return avg_loss, float("nan"), 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0, 0.5, None

    # Convert each tensor to numpy and ravel to ensure 1-D arrays before concatenation
    all_probs_np = np.concatenate([safe_to_numpy(t).ravel() for t in all_probs], axis=0)
    all_labels_np = np.concatenate([safe_to_numpy(t).ravel() for t in all_labels], axis=0).astype(int)

    # AUROC on continuous probs
    if len(np.unique(all_labels_np)) > 1:
        auroc = float(roc_auc_score(all_labels_np, all_probs_np))
    else:
        auroc = float("nan")

    # compute classic metrics and PR-AUC / best F1 threshold
    auc_val, f1_05, prec_05, rec_05, pr_auc, best_f1, best_thresh = compute_classic_metrics(all_labels_np, all_probs_np)

    # confusion matrix at 0.5
    preds_05 = (all_probs_np > 0.5).astype(int)
    try:
        cm = sk_confusion_matrix(all_labels_np, preds_05)
    except Exception:
        cm = None

    gate_info = None
    if len(gate_vals) > 0:
        gate_info = np.concatenate([safe_to_numpy(g).reshape(g.shape[0], -1) for g in gate_vals], axis=0)

    return avg_loss, auroc, f1_05, prec_05, rec_05, pr_auc, best_f1, best_thresh, cm, all_probs_np, all_labels_np, gate_info

# -------------------------
# Cross-validation training
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

        # Build imbalance strategy
        train_labels_arr = np.array([int(dataset[i].y.item()) for i in train_idx])
        num_pos = int((train_labels_arr == 1).sum())
        num_neg = int((train_labels_arr == 0).sum())
        print(f"Fold {fold+1} pos/neg = {num_pos}/{num_neg}")

        if args.imbalance_strategy == "sampler":
            class_counts = np.bincount(train_labels_arr)
            class_counts = np.maximum(class_counts, 1)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_labels_arr]
            sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float),
                                            num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers)
            pos_weight = torch.tensor([1.0], device=device)
            print("Using WeightedRandomSampler (oversampling); pos_weight=1.0")
        elif args.imbalance_strategy == "pos_weight":
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            if num_pos == 0:
                pos_w = 1.0
            else:
                pos_w = float(num_neg) / float(max(1, num_pos))
                pos_w = float(np.sqrt(pos_w))  # dampened ratio
            pos_weight = torch.tensor([pos_w], dtype=torch.float32).to(device)
            print(f"Using pos_weight={pos_weight.item():.4f}")
        else:
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            pos_weight = torch.tensor([1.0], device=device)
            print("No imbalance handling selected; using plain sampler and pos_weight=1.0")

        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # instantiate model
        model = get_torch_model(model_name, sample_x, sample_y, sample_patch_centers).to(device)
        # use BCEWithLogitsLoss and apply pos_weight if requested
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_val_auc = -float("inf")
        ckpt_dir = "Checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold+1}_best.pth")

        # per-epoch collections for plotting
        fold_train_losses, fold_val_losses = [], []
        fold_train_aucs, fold_val_aucs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        fold_val_pr_aucs, fold_val_bestf1s = [], []

        for epoch in range(1, args.num_epochs + 1):
            train_loss, train_auroc, train_f1, train_prec, train_rec, train_pr_auc, train_best_f1, train_best_thresh, train_cm, _, _, train_gate = run_epoch(train_loader, model, criterion, optimizer, train=True)
            val_loss, val_auroc, val_f1, val_prec, val_rec, val_pr_auc, val_best_f1, val_best_thresh, val_cm, val_probs_np, val_labels_np, val_gate = run_epoch(val_loader, model, criterion, optimizer=None, train=False)

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_aucs.append(train_auroc)
            fold_val_aucs.append(val_auroc)
            fold_train_f1s.append(train_f1)
            fold_val_f1s.append(val_f1)
            fold_val_pr_aucs.append(val_pr_auc)
            fold_val_bestf1s.append(val_best_f1)

            # print epoch summary including PR-AUC and best threshold
            print(f"Epoch {epoch}/{args.num_epochs} | TrainLoss {train_loss:.4f} TrainAUC {train_auroc:.4f} TrainF1 {train_f1:.4f} | "
                  f"ValLoss {val_loss:.4f} ValAUC {val_auroc:.4f} ValF1@0.5 {val_f1:.4f} PR-AUC {val_pr_auc:.4f} BestF1 {val_best_f1:.4f}@{val_best_thresh:.3f}")

            # log gate stats if available
            if val_gate is not None:
                try:
                    print(f"Gate stats - mean: {float(np.mean(val_gate)):.4f}, std: {float(np.std(val_gate)):.4f}")
                except Exception:
                    pass

            # checkpoint by val_auc
            if not np.isnan(val_auroc) and val_auroc > best_val_auc:
                best_val_auc = val_auroc
                torch.save({"model_state_dict": model.state_dict(), "val_auc": float(val_auroc)}, ckpt_path)

        # after epochs: load best checkpoint and compute final metrics on val set
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        # final evaluation on val set
        model.eval()
        val_preds = []
        val_labels = []
        val_gates = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                if isinstance(out, tuple):
                    logits, info = out
                else:
                    logits, info = out, {}
                logits = logits.view(-1, 1)
                # keep 1-D arrays even for batch-size=1
                probs = torch.sigmoid(to_logits_if_probs(logits)).cpu().view(-1)   # (N,)
                labels = batch.y.detach().cpu().view(-1).int()                    # (N,)
                val_preds.append(probs)
                val_labels.append(labels)
                if isinstance(info, dict) and 'gate' in info and info['gate'] is not None:
                    val_gates.append(info['gate'].detach().cpu().view(info['gate'].shape[0], -1))
        # concatenate safely
        val_preds_np = np.concatenate([safe_to_numpy(t).ravel() for t in val_preds], axis=0)
        val_labels_np = np.concatenate([safe_to_numpy(t).ravel() for t in val_labels], axis=0).astype(int)
        # handle gates if present
        if len(val_gates) > 0:
            val_gates_np = np.concatenate([safe_to_numpy(g) for g in val_gates], axis=0)
        else:
            val_gates_np = None

        val_preds_np = np.concatenate([safe_to_numpy(t) for t in val_preds], axis=0)
        val_labels_np = np.concatenate([safe_to_numpy(t) for t in val_labels], axis=0).astype(int)

        val_auc, val_f1_05, val_prec_05, val_rec_05, val_pr_auc, val_best_f1, val_best_thresh = compute_classic_metrics(val_labels_np, val_preds_np)

        print(f"[Fold {fold+1}] Final (best-checkpoint) metrics: AUC: {val_auc:.4f} F1@0.5: {val_f1_05:.4f} Prec: {val_prec_05:.4f} Rec: {val_rec_05:.4f} PR-AUC: {val_pr_auc:.4f} BestF1: {val_best_f1:.4f}@{val_best_thresh:.3f}")

        # save val predictions for inspection
        np.savez(os.path.join(ckpt_dir, f"{model_name}_fold{fold+1}_val_preds.npz"),
                 probs=val_preds_np, labels=val_labels_np, best_thresh=val_best_thresh)

        all_fold_metrics.append({"auc": val_auc, "f1": val_f1_05, "prec": val_prec_05, "rec": val_rec_05, "pr_auc": val_pr_auc, "best_f1": val_best_f1})

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
        auc, f1, prec, rec, pr_auc, best_f1, best_thresh = compute_classic_metrics(y_val, probs)
        print(f"[Fold {fold+1}] metrics: AUC {auc:.4f} F1 {f1:.4f} Prec {prec:.4f} Rec {rec:.4f} PR-AUC {pr_auc:.4f} BestF1 {best_f1:.4f}@{best_thresh:.3f}")
        model_dir = "Models_sklearn"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, f"{model_name}_fold{fold+1}.joblib"))
        all_fold_metrics.append({"auc": auc, "f1": f1, "prec": prec, "rec": rec, "pr_auc": pr_auc, "best_f1": best_f1})
    return all_fold_metrics


# -------------------------
# Run chosen model flow
# -------------------------
if args.model in ("gatedfusion_graphtransform", "graphtransformer_performer", "graphtransformer_mha", "mlp_pt"):
    fold_metrics = train_torch_model_cv(args.model)
else:
    fold_metrics = train_sklearn_model_cv(args.model)


# Aggregate and print summary (mean ± std)
def summarize_fold_metrics(fold_metrics):
    keys = ["auc", "f1", "prec", "rec", "pr_auc", "best_f1"]
    out = {}
    for k in keys:
        vals = np.array([m.get(k, float("nan")) for m in fold_metrics], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
        out[f"{k}_std"] = float(np.nanstd(vals)) if vals.size > 0 else float("nan")
    return out


summary = summarize_fold_metrics(fold_metrics)
print("Summary (mean ± std across folds):")
print(f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
print(f"F1 : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
print(f"Prec: {summary['prec_mean']:.4f} ± {summary['prec_std']:.4f}")
print(f"Rec : {summary['rec_mean']:.4f} ± {summary['rec_std']:.4f}")
print(f"PR-AUC: {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}")
print(f"BestF1: {summary['best_f1_mean']:.4f} ± {summary['best_f1_std']:.4f}")