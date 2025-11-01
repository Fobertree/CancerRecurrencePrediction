"""
train_graph_transformer_from_metadata.py
----------------------------------------

End-to-end graph-transformer training pipeline that reads labels from new_metadata.csv.

Assumptions:
 - Patches or per-slide features live under "dinov2_patches/<svs_name>/"
 - The CSV file `new_metadata.csv` includes at least:
       svs_name, Oncotype DX Breast Recurrence Score
 - You can choose to treat Oncotype scores as regression or classification targets.

Usage example:
    python train_graph_transformer_from_metadata.py \
        --patch_dir dinov2_patches \
        --metadata_csv new_metadata.csv \
        --epochs 30 --batch_size 8 --task classification
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from PIL import Image
import timm

# ---------------------------------------
# Step 1. Read metadata (svs_name + label)
# ---------------------------------------

def load_slide_labels(metadata_csv, task="classification"):
    """
    Reads metadata CSV, extracts svs_name and Oncotype DX Breast Recurrence Score.
    If task == 'classification', bins numeric scores into 3 classes:
        0 = low (<18), 1 = intermediate (18-30), 2 = high (>30)
    Returns dict: {svs_name: label_value}
    """
    df = pd.read_csv(metadata_csv)
    if "svs_name" not in df.columns or "Oncotype DX Breast Recurrence Score" not in df.columns:
        raise ValueError("CSV must contain 'svs_name' and 'Oncotype DX Breast Recurrence Score' columns")

    df = df.dropna(subset=["svs_name", "Oncotype DX Breast Recurrence Score"])
    svs_names = df["svs_name"].astype(str).str.replace(".svs", "", regex=False)
    scores = df["Oncotype DX Breast Recurrence Score"].astype(float)

    if task == "classification":
        # Bin the scores according to standard Oncotype DX ranges
        bins = [-np.inf, 18, 30, np.inf]
        labels = [0, 1, 2]  # 0=low,1=intermediate,2=high
        binned = np.digitize(scores, bins) - 1
        mapping = dict(zip(svs_names, binned))
    else:
        # Regression
        mapping = dict(zip(svs_names, scores))

    print(f"Loaded {len(mapping)} slide labels from {metadata_csv}")
    return mapping


# --------------------------------
# Step 2. Patch feature extraction
# --------------------------------

def get_backbone(name="resnet50", pretrained=True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool='avg')
    model.eval()
    return model

@torch.no_grad()
def extract_patch_features(patch_dir, out_dir, backbone_name="resnet50", device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_backbone(backbone_name).to(device)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list(Path(patch_dir).rglob("*.png"))
    if not paths:
        raise RuntimeError(f"No patches found under {patch_dir}")

    # expect layout: dinov2_patches/<svs_name>/*.png
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # group by slide folder
    slides = sorted({p.parent.name for p in paths})
    for slide_id in tqdm(slides, desc="extract_features"):
        slide_dir = Path(patch_dir) / slide_id
        out_slide = out_root / slide_id
        out_slide.mkdir(parents=True, exist_ok=True)
        imgs = list(slide_dir.glob("*.png"))
        feats, coords = [], []
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            parts = img_path.stem.split("_")
            x, y = (0, 0)
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                x, y = int(parts[-2]), int(parts[-1])
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy().squeeze()
            feats.append(feat)
            coords.append([x, y])
        np.save(out_slide / "node_feats.npy", np.stack(feats))
        np.save(out_slide / "coords.npy", np.stack(coords))


# --------------------------------
# Step 3. Build graphs (k-NN edges)
# --------------------------------

def build_graphs_from_features(feat_root, out_graph_root, k=8):
    feat_root = Path(feat_root)
    out_root = Path(out_graph_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slides = [d for d in feat_root.iterdir() if d.is_dir()]
    for slide in tqdm(slides, desc="build_graphs"):
        feats = np.load(slide / "node_feats.npy")
        coords = np.load(slide / "coords.npy")
        N = feats.shape[0]
        if N < 2: continue
        nbrs = NearestNeighbors(n_neighbors=min(k+1, N)).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        edges = []
        for i in range(N):
            for j in indices[i,1:]:
                edges.append((i, j))
                edges.append((j, i))
        edges = np.array(edges, dtype=np.int32)
        out_slide = out_root / slide.name
        out_slide.mkdir(parents=True, exist_ok=True)
        np.save(out_slide / "node_feats.npy", feats)
        np.save(out_slide / "coords.npy", coords)
        np.save(out_slide / "edges.npy", edges)


# --------------------------------
# Step 4. Dataset + collate
# --------------------------------

class SlideGraphDataset(Dataset):
    def __init__(self, graphs_root, label_dict, task="classification", max_nodes=2000):
        self.root = Path(graphs_root)
        self.slide_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.labels = label_dict
        self.task = task
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.slide_dirs)

    def __getitem__(self, idx):
        d = self.slide_dirs[idx]
        feats = np.load(d / "node_feats.npy")
        coords = np.load(d / "coords.npy")
        edges = np.load(d / "edges.npy") if (d / "edges.npy").exists() else np.zeros((0,2),dtype=np.int32)
        if feats.shape[0] > self.max_nodes:
            keep = np.random.choice(feats.shape[0], self.max_nodes, replace=False)
            feats = feats[keep]
            coords = coords[keep]
        slide_name = d.name
        label = self.labels.get(slide_name, None)
        if label is None:
            label = -1
        sample = {
            "slide": slide_name,
            "feats": torch.from_numpy(feats).float(),
            "coords": torch.from_numpy(coords).float(),
            "edges": torch.from_numpy(edges).long(),
            "label": torch.tensor(label, dtype=torch.float if self.task=="regression" else torch.long)
        }
        return sample

def collate_graphs(batch):
    B = len(batch)
    D = batch[0]["feats"].shape[1]
    Ns = [b["feats"].shape[0] for b in batch]
    Nmax = max(Ns)
    feats = torch.zeros((B, Nmax, D))
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    labels = []
    for i, b in enumerate(batch):
        n = b["feats"].shape[0]
        feats[i,:n] = b["feats"]
        mask[i,:n] = 1
        labels.append(b["label"])
    labels = torch.stack(labels)
    return {"feats": feats, "mask": mask, "labels": labels}


# --------------------------------
# Step 5. Model (Graph Transformer)
# --------------------------------

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_heads=4, n_layers=3, out_dim=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, feats, mask):
        pad_mask = ~mask
        x = self.proj(feats)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(x)


# --------------------------------
# Step 6. Training Loop
# --------------------------------

def train_model(graphs_root, label_dict, out_dir, task="classification", epochs=20, batch_size=4, lr=1e-4):
    dataset = SlideGraphDataset(graphs_root, label_dict, task)
    n = len(dataset)
    idx = list(range(n)); random.shuffle(idx)
    split = int(n*0.8)
    train_ds = Subset(dataset, idx[:split])
    val_ds   = Subset(dataset, idx[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    in_dim = dataset[0]["feats"].shape[1]
    if task == "classification":
        out_dim = len(set(label_dict.values()))
        model = GraphTransformer(in_dim, out_dim=out_dim)
        criterion = nn.CrossEntropyLoss()
    else:
        model = GraphTransformer(in_dim, out_dim=1)
        criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf") if task=="regression" else 0.0

    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            feats = batch["feats"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(feats, mask).squeeze()
            if task == "classification":
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            if task == "classification":
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / len(train_ds)
        if task == "classification":
            train_acc = correct / total
            print(f"Epoch {epoch}: Train loss {train_loss:.4f} acc {train_acc:.4f}")
        else:
            print(f"Epoch {epoch}: Train MSE {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, vtotal = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["feats"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(feats, mask).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                if task == "classification":
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    vtotal += labels.size(0)

        val_loss /= len(val_ds)
        if task == "classification":
            val_acc = val_correct / vtotal
            print(f"Epoch {epoch}: Val loss {val_loss:.4f}, acc {val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
                print("Saved new best model.")
        else:
            print(f"Epoch {epoch}: Val MSE {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
                print("Saved new best model.")


# --------------------------------
# CLI Entry
# --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", type=str, default="dinov2_patches")
    parser.add_argument("--metadata_csv", type=str, default="new_metadata.csv")
    parser.add_argument("--intermediate_feats", type=str, default="feat_graphs")
    parser.add_argument("--graphs_root", type=str, default="graphs")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--task", type=str, default="classification", choices=["classification","regression"])
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    labels = load_slide_labels(args.metadata_csv, task=args.task)

    # 1. Extract features
    extract_patch_features(args.patch_dir, args.intermediate_feats)

    # 2. Build graphs
    build_graphs_from_features(args.intermediate_feats, args.graphs_root, k=args.k)

    # 3. Train model
    train_model(args.graphs_root, labels, args.out, task=args.task,
                epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
