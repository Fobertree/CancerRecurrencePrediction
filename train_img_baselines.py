# train_img_baselines.py
import os, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

from torchmetrics.classification import BinaryConfusionMatrix, BinaryPrecision, BinaryRecall

# -----------------------------
# Config
# -----------------------------
METADATA_PATH  = "new_metadata(in).csv"         # your full metadata file
IMAGES_DIR     = "PreprocessedData"         # where JPEGs live
K_FOLDS        = 3
NUM_EPOCHS     = 20
BATCH_SIZE     = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-5
NUM_WORKERS    = 2 if os.name != "nt" else 0  # Windows safer with 0
MODEL_NAME     = "vit_b_16"                 # "vit_b_16" or "resnet50"
IMG_SIZE       = 224                        # ViT default is 224; larger can help WSIs
MIXED_PRECISION= True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}")

# -----------------------------
# Dataset
# -----------------------------
class ImageCSV(Dataset):
    def __init__(self, df, train=True, img_size=448):
        self.df = df.reset_index(drop=True)
        # Stronger augs for train; light for val
        if train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row.label, dtype=torch.float32)
        return x, y

# -----------------------------
# Model factory
# -----------------------------
# -----------------------------
# Model factory
# -----------------------------
def build_model(name: str, out_dim=1, freeze_backbone=True):
    if name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, out_dim)  # this head stays trainable by default

    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if freeze_backbone:
            # freeze everything first
            for p in m.parameters():
                p.requires_grad = False

        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, out_dim)  # new classifier layer

        # ensure classifier is trainable
        for p in m.fc.parameters():
            p.requires_grad = True

    else:
        raise ValueError("Unknown MODEL_NAME")
    return m

# -----------------------------
# Train/val epoch
# -----------------------------
def run_epoch(loader, model, criterion, optimizer=None):
    train = optimizer is not None
    model.train(mode=train)

    loss_sum = 0.0
    preds_all, labels_all = [], []

    precision_metric = BinaryPrecision().to(DEVICE)
    recall_metric    = BinaryRecall().to(DEVICE)
    cm_metric        = BinaryConfusionMatrix().to(DEVICE)

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION and train)

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).view(-1,1)

        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
            logits = model(xb)              # (B,1)
            loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_sum += loss.item() * xb.size(0)
        probs = torch.sigmoid(logits.detach())
        preds_all.append(probs.cpu())
        labels_all.append(yb.detach().cpu())

        # metrics expect probs (float) and labels (0/1)
        cm_metric.update(probs, yb)
        precision_metric.update(probs, yb)
        recall_metric.update(probs, yb)

    avg_loss = loss_sum / len(loader.dataset)
    preds = torch.cat(preds_all).numpy()
    labels= torch.cat(labels_all).numpy()
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, preds)
    else:
        auroc = np.nan
    f1 = f1_score(labels, (preds > 0.5).astype(np.int32), zero_division=0)

    cm        = cm_metric.compute().cpu()
    precision = precision_metric.compute().cpu()
    recall    = recall_metric.compute().cpu()

    return avg_loss, auroc, f1, cm, precision, recall

# -----------------------------
# Main K-fold
# -----------------------------
def main():
    df = pd.read_csv(METADATA_PATH)

    if "label" not in df.columns and "Oncotype DX Breast Recurrence Score" in df.columns:
        df = df.rename(columns={"Oncotype DX Breast Recurrence Score": "label"})

    df["path"] = df["svs_name"].apply(
        lambda s: os.path.join(IMAGES_DIR, f"{s}.jpeg")
    )

    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    print("Total rows after filtering for existing JPEGs:", len(df))
    print(df[["svs_name", "label", "path"]].head())

    y = df["label"].values.astype(int)


    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    all_train_losses, all_val_losses = [], []
    all_train_aurocs, all_val_aurocs = [], []
    all_train_f1s,   all_val_f1s   = [], []
    all_train_prec,  all_val_prec  = [], []
    all_train_rec,   all_val_rec   = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
        print(f"\n=== Fold {fold}/{K_FOLDS} ===")
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]

        # Sampler for class balance
        class_counts = np.bincount(df_tr["label"].values.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[df_tr["label"].values.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True
        )

        ds_tr = ImageCSV(df_tr, train=True,  img_size=IMG_SIZE)
        ds_va = ImageCSV(df_va, train=False, img_size=IMG_SIZE)

        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
        dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))

        model = build_model(MODEL_NAME, out_dim=1, freeze_backbone=True).to(DEVICE)

        # pos_weight from current train split
        n_pos = int((df_tr["label"]==1).sum())
        n_neg = int((df_tr["label"]==0).sum())
        pos_w = torch.tensor([max(n_neg,1)/max(n_pos,1)], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)

        fold_train_losses, fold_val_losses = [], []
        fold_train_aurocs, fold_val_aurocs = [], []
        fold_train_f1s,   fold_val_f1s   = [], []
        fold_train_prec,  fold_val_prec  = [], []
        fold_train_rec,   fold_val_rec   = [], []

        best_val = math.inf
        patience = 5
        bad = 0

        for epoch in range(1, NUM_EPOCHS+1):
            tr_loss, tr_auc, tr_f1, tr_cm, tr_prec, tr_rec = run_epoch(dl_tr, model, criterion, optimizer)
            va_loss, va_auc, va_f1, va_cm, va_prec, va_rec = run_epoch(dl_va, model, criterion, optimizer=None)

            scheduler.step(va_loss)

            fold_train_losses.append(tr_loss); fold_val_losses.append(va_loss)
            fold_train_aurocs.append(tr_auc);  fold_val_aurocs.append(va_auc)
            fold_train_f1s.append(tr_f1);      fold_val_f1s.append(va_f1)
            fold_train_prec.append(tr_prec);   fold_val_prec.append(va_prec)
            fold_train_rec.append(tr_rec);     fold_val_rec.append(va_rec)

            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                  f"Train Loss {tr_loss:.4f} AUROC {tr_auc:.4f} F1 {tr_f1:.4f} | "
                  f"Val Loss {va_loss:.4f} AUROC {va_auc:.4f} F1 {va_f1:.4f} "
                  f"P {va_prec:.4f} R {va_rec:.4f}")

            # simple early stop on val loss
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stop at epoch {epoch}")
                    break

        all_train_losses.append(fold_train_losses); all_val_losses.append(fold_val_losses)
        all_train_aurocs.append(fold_train_aurocs); all_val_aurocs.append(fold_val_aurocs)
        all_train_f1s.append(fold_train_f1s);       all_val_f1s.append(fold_val_f1s)
        all_train_prec.append(fold_train_prec);     all_val_prec.append(fold_val_prec)
        all_train_rec.append(fold_train_rec);       all_val_rec.append(fold_val_rec)

        print(f"Fold {fold} summary | "
              f"Val AUROC {np.nanmean(fold_val_aurocs):.4f} F1 {np.nanmean(fold_val_f1s):.4f}")

    print("\n=== Overall (last epoch per fold) ===")
    print(f"AUROC: {np.nanmean([np.nanmean(v) for v in all_val_aurocs]):.4f} | "
          f"F1: {np.nanmean([np.nanmean(v) for v in all_val_f1s]):.4f}")

if __name__ == "__main__":
    main()
