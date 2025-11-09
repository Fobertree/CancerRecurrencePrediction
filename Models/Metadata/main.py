import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from NaiveMLP import NaiveMLP

# replace this with your metadata path
METADATA_PATH = "/Users/alexanderliu/EmoryCS/CancerRecurrencePrediction/new_metadata.csv"


# DF LOAD + PREPROCESS
def load_metadata_features():
    df = pd.read_csv(METADATA_PATH)
    # drop first to prevent multicollinearity
    df = pd.get_dummies(df, columns=["HistologicType"], drop_first=True)

    continuous_cols = ['Age', 'TumorSize']

    # Initialize KBinsDiscretizer for 4 bins using 'quantile' strategy and 'ordinal' encoding
    n_bins = 4
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    # Apply discretization to the selected columns
    df_discretized_values = discretizer.fit_transform(df[continuous_cols])

    # Create a new DataFrame with the discretized columns
    df_discretized = pd.DataFrame(df_discretized_values, columns=[col + '_binned' for col in continuous_cols])

    # Combine with original non-discretized columns (e.g., categorical_col)
    df = pd.concat([df.drop(columns=continuous_cols), df_discretized], axis=1)
    return df

df = load_metadata_features()

# END PREPROCESS

if __name__ == "__main__":
    X_cols = [col for col in df.columns if col not in ["Oncotype DX Breast Recurrence Score", "svs_name"]]
    X = df[X_cols].values
    y = df["Oncotype DX Breast Recurrence Score"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to tensors - stratified holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([num_neg/num_pos], dtype=torch.float32).to(device)

    # MODEL
    model = NaiveMLP(in_channels=X_train.shape[1], hidden=64).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.4f}")


    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy().flatten()

    # Threshold at 0.5
    pred_labels = (preds >= 0.5).astype(int)
    y_true = y_test.numpy().flatten()

    f1 = f1_score(y_true, pred_labels)
    auc = roc_auc_score(y_true, preds)

    print("\n=== Evaluation Results ===")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")