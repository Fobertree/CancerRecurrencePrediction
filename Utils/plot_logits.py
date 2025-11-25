import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def plot_logits(data_tensor, labels = None, summary_stats = False):
    data_np = data_tensor.squeeze().detach().cpu().numpy()
    df = pd.DataFrame({'Value': data_np, 'Category': labels})

    if summary_stats:
        tensor_summary_stats(data_tensor)

    try:
        plt.figure(figsize=(8, 6)) # Optional: set figure size
        sns.histplot(df, x="Value",kde=True, binwidth=0.1, binrange=(0,1), hue= "Category", multiple="stack")
        plt.title('Seaborn Histogram of PyTorch Tensor Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig("logit_dist.png")
    except Exception as e:
        print(e)

    plt.close()

def tensor_summary_stats(tensor):
    print(f"Mean: {torch.mean(tensor)}")
    print(f"Standard Deviation: {torch.std(tensor)}")
    print(f"Minimum value: {torch.min(tensor)}")
    print(f"Maximum value: {torch.max(tensor)}")