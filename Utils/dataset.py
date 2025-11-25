import torch
# if CPU RAM cannot handle the dataset, use regular Dataset instead
from torch_geometric.data import InMemoryDataset, download_url
import torch_geometric.data as pyg_data

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# --- Safe unpickling for PyTorch ≥ 2.6 ---
if hasattr(torch.serialization, "add_safe_globals"):
    allowlist = [pyg_data.Data]
    # Add DataEdgeAttr if it exists in your PyG version
    if hasattr(pyg_data, "DataEdgeAttr"):
        allowlist.append(pyg_data.DataEdgeAttr)
    torch.serialization.add_safe_globals(allowlist)

class CancerRecurrenceGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset that loads precomputed WSI graphs
    from a directory. Supports spatial, similarity, and combined graphs.
    """
    def __init__(self, root, graph_type="combined", transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Path to the folder containing saved graphs (from graphbuilder.py)
            graph_type (str): "spat", "sim", or "combined"
        """
        self.graph_type = graph_type.lower()
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load all graphs
        if not os.path.exists(self.processed_paths[0]):
            # WARNING THIS IS DANGEROUS
            # If we change pipeline it won't run unless we delete processed folder
            self.process()  # process if not already done
        self.data, self.slices = torch.load(self.processed_paths[0], map_location="cpu", weights_only=False)


    @property
    def raw_file_names(self):
        # Not used in this dataset; graphs are already precomputed
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Already processed externally
        pass

    def process(self):
        """
        Read saved graphs from root directory, attach labels, and filter/transform if needed.
        """
        label_csv = "data/new_metadata.csv"
        if not os.path.exists(label_csv):
            raise FileNotFoundError(f"Missing label file: {label_csv}")
        
        print("PROCESSING DATASET")

        labels_df = pd.read_csv(label_csv)
        print(f"Loaded {len(labels_df)} labels from metadata")

        X_df = load_metadata_features(labels_df)
        print(X_df)

        # Convert to dict for fast lookup (e.g., {slide_id: label})
        label_dict = dict(zip(labels_df["svs_name"], labels_df["Oncotype DX Breast Recurrence Score"]))
        X_dict = dict(zip(labels_df["svs_name"], X_df.values))

        print(X_dict)

        graph_files = [
            f for f in os.listdir(self.root)
            if f.startswith(f"{self.graph_type}") and f.endswith(".pt")
        ]
        graph_files.sort()  # ensure consistent order

        data_list = []
        for f in graph_files:
            graph_path = os.path.join(self.root, f)
            data = torch.load(graph_path, map_location="cpu", weights_only=False)

            # Extract slide_id from filename (adapt this pattern to match yours)
            slide_id = f.replace(f"{self.graph_type}_", "").replace(".pt", "")
            if slide_id not in label_dict:
                print(f"Warning: no label for {slide_id}, skipping")
                continue

            data.y = torch.tensor([int(label_dict[slide_id])], dtype=torch.long)

            # metadata features
            data.metadata = torch.tensor([X_dict[slide_id]], dtype=torch.float)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError(f"No labeled graphs found for type '{self.graph_type}' in {self.root}")

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_metadata_features(df):
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

    X_cols = [col for col in df.columns if col not in ["Oncotype DX Breast Recurrence Score", "svs_name"]]
    X_df = df[X_cols]

    return X_df
