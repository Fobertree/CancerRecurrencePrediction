import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve
from torch_geometric.nn import HypergraphConv
from sklearn.neighbors import NearestNeighbors
import numpy as np

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
METADATA_PATH = "C:/Users/thoma/Documents/CS-371W/CancerRecurrencePrediction/new_metadata.csv"

def load_metadata_features():
    df = pd.read_csv(METADATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # One-hot encode categorical features
    categorical_cols = ["HistologicType"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scale continuous features
    continuous_cols = ['Age', 'TumorSize']
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    
    return df

df = load_metadata_features()

# Target
y = df["Oncotype DX Breast Recurrence Score"].values
X_cols = [col for col in df.columns if col not in ["Oncotype DX Breast Recurrence Score", "svs_name"]]
X = df[X_cols].values.astype(np.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# kNN HYPERGRAPH CONSTRUCTION
# -------------------------------
def build_knn_hypergraph(X_tensor, k=5):
    X_np = X_tensor.numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)
    
    hyperedges = []
    n_nodes = X_np.shape[0]
    
    # Each node + its neighbors forms a hyperedge
    for i in range(n_nodes):
        edge_nodes = indices[i]  # includes self
        hyperedges.append(edge_nodes.tolist())
    
    # Convert hyperedges to edge_index
    edge_index = []
    for h_idx, nodes in enumerate(hyperedges):
        for node in nodes:
            edge_index.append([node, h_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

train_edge_index = build_knn_hypergraph(X_train, k=5)
test_edge_index = build_knn_hypergraph(X_test, k=5)

# -------------------------------
# HGNN MODEL
# -------------------------------
class HypergraphNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, dropout=0.2):
        super(HypergraphNN, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.conv3 = HypergraphConv(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

model = HypergraphNN(in_channels=X_train.shape[1]).to(device)

# -------------------------------
# FOCAL LOSS
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            F_loss = self.alpha * F_loss
        return F_loss.mean()

criterion = FocalLoss(gamma=2.0)

# -------------------------------
# TRAINING
# -------------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device), train_edge_index.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# -------------------------------
# EVALUATION
# -------------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device), test_edge_index.to(device)).cpu().numpy().flatten()

# Threshold tuning via F1
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test.numpy().flatten(), preds)
f1_scores = 2*precision_vals*recall_vals/(precision_vals+recall_vals+1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
pred_labels = (preds >= best_thresh).astype(int)

y_true = y_test.numpy().flatten()
f1 = f1_score(y_true, pred_labels)
auc = roc_auc_score(y_true, preds)
precision = precision_score(y_true, pred_labels, zero_division=0)
recall = recall_score(y_true, pred_labels, zero_division=0)

print("\n=== HGNN Evaluation Results ===")
print(f"Best Threshold: {best_thresh:.3f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
