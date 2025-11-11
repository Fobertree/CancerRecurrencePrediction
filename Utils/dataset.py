import torch
# if CPU RAM cannot handle the dataset, use regular Dataset instead
from torch_geometric.data import InMemoryDataset, download_url
import torch_geometric.data as pyg_data

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd

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

        # Convert to dict for fast lookup (e.g., {slide_id: label})
        label_dict = dict(zip(labels_df["svs_name"], labels_df["Oncotype DX Breast Recurrence Score"]))

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

