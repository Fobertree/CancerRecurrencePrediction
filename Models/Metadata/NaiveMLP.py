import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

class NaiveMLP(nn.Module):
    def __init__(self, in_channels, hidden, out_channels=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_channels), 
            nn.Sigmoid() # for final binary classification
        )
    
    def forward(self, x):
        return self.seq(x)
    

# k bins discretizer