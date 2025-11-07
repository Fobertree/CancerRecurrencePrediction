import torch

model = ...

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# weighted bce for class imbalance
criterion = torch.nn.BCELoss(weight=0.2) # TODO: change weight parameter


if __name__ == "__main__":
    print("main")