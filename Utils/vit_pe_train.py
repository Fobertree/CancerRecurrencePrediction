from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import pandas as pd
from tqdm import tqdm
from torchmetrics.classification import confusion_matrix, BinaryPrecision, BinaryRecall
from sklearn.metrics import roc_auc_score, f1_score

class ImagePEDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.paths = [os.path.join("Image", f"{im}.jpeg") for im in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        lbl = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return img, lbl

class DINOv2ForBinaryClassification(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.backbone = pretrained_model  # ViT-S/14
        embed_dim = pretrained_model.embed_dim  # 384 for vits14

        # Small head — good for fine-tuning
        self.classifier = nn.Linear(embed_dim, 1)  # binary logit

    def forward(self, x):
        feats = self.backbone(x)          # (B, 384)
        logits = self.classifier(feats)   # (B, 1)
        return logits

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( # ImageNet norm stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
    model = DINOv2ForBinaryClassification(dinov2_model)

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])

    metadata_df = pd.read_csv('Data/new_metadata.csv', index_col=0).set_index('svs_name')

    image_paths = [f.removesuffix(".jpeg").removesuffix(".png")
        for root, _, files in os.walk("Image")
        for f in files
        if f.lower().endswith((".png", ".jpeg")) and f.removesuffix(".jpeg").removesuffix(".png") in metadata_df.index]

    print(len(image_paths))

    labels = metadata_df.loc[image_paths]["Oncotype DX Breast Recurrence Score"]
    dataset = ImagePEDataset(image_paths, labels, transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    # logger.info(f"Class ratio: Pos: {num_pos}, Neg: {num_neg}")
    print(f"Class ratio: Pos: {num_pos}, Neg: {num_neg}")
    weight = (num_neg / (num_pos)) # weight is hyperparam

    pos_weight = torch.tensor([weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    model.train()
    for epoch in tqdm(range(10), desc="Outer Epoch Loop"):
        for imgs, labels in tqdm(train_loader, "Inner Batch Loop"):
            labels = labels.unsqueeze(1)  # -> (B,1)

            logits = model(imgs)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in train_loader: # no separate test lader
            # don't worry abt data leakage for PE training
            logits = model(imgs)  # (B,1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.append(preds)
            all_labels.append(labels)

    torch.save(model.backbone.state_dict(), "fine-tuned-dino-S14.pt")

    all_preds_tensor = torch.squeeze(torch.cat(all_preds, dim=0).int())
    all_labels_tensor = torch.squeeze(torch.cat(all_labels, dim=0).int())

    auroc = roc_auc_score(all_labels_tensor, all_preds_tensor)
    f1 = f1_score(all_labels_tensor, all_preds_tensor > 0.5, zero_division=0)
    cm_metric = confusion_matrix.BinaryConfusionMatrix()
    cm_metric.update(all_preds_tensor, all_labels_tensor)

    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    precision_metric.update(all_preds_tensor, all_labels_tensor)
    recall_metric.update(all_preds_tensor, all_labels_tensor)

    cm = cm_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()

    print(auroc, f1, cm, precision, recall)