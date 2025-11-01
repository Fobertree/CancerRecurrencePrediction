import torch
# if CPU RAM cannot handle the dataset, use regular Dataset instead
from torch_geometric.data import InMemoryDataset, download_url

# TODO: implement this and call constructor in graphbuilder.py
# class CancerRecurrenceGraphDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.load(self.processed_paths[0])
    
#     @property
#     def raw_file_names(self):
#         return ['some_file_1', 'some_file_2', ...]

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         pass

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = [...]

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         self.save(data_list, self.processed_paths[0])


import os
import torch
from torch_geometric.data import InMemoryDataset, Data

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
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        Read saved graphs from root directory and filter/transform if needed.
        """
        graph_files = [
            f for f in os.listdir(self.root)
            if f.endswith(f"{self.graph_type}_graph") or f.endswith(f"{self.graph_type}_graph.pt")
        ]
        graph_files.sort()  # ensure consistent order

        data_list = []
        for f in graph_files:
            graph_path = os.path.join(self.root, f)
            data = torch.load(graph_path)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError(f"No graphs found for type '{self.graph_type}' in {self.root}")

        # Use PyG utility to collate all graphs
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

