# %%
# %pip install -r requirements.txt
# %pip install pandas

# %%
print("main ipynb")

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import importlib
import Utils.wsi as wsi

# importlib.reload(Utils.wsi)

print("Output: ", wsi.load_wsi("data", "data/metadata.csv"))

# %%
# %pip install openslide-bin

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("OK")
exit()
# Path to your CSV file
csv_path = "data/Metadata.csv"

# Load data
df = pd.read_csv(csv_path)

# Drop rows with missing values (optional, or use imputation)
df = df.dropna()

df['Oncotype DX Breast Recurrence Score'] = (df['Oncotype DX Breast Recurrence Score'] > 25).astype(int)

# Example: target column = 'Label' (0 = control, 1 = disease)
target_col = 'Oncotype DX Breast Recurrence Score'

df.to_csv('data/new_metadata.csv')

df = df.drop(columns=['svs_name', 'id'], errors='ignore')

# Split features and labels
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical features automatically
X = pd.get_dummies(X, drop_first=True)

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class ClinicalDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ClinicalDataset(X_train, y_train)
test_dataset = ClinicalDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(X_train.shape[1])

criterion = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_label = (y_pred >= 0.5).float()

accuracy = (y_pred_label == y_test).float().mean()
print(f"Test Accuracy: {accuracy.item():.4f}")


# %%



