from Utils.wsi import load_wsi

# Path to raw WSI files
wsi_dir = "data"
metadata_path = "data/new_metadata.csv"
output_patch_dir = "dinov2_patches"

wsi_loader_out = load_wsi(
    directory_path=wsi_dir,
    metadata_path=metadata_path,
    output_dir=output_patch_dir,
    threshold=None,        # process all WSIs
    patch_size=256,
    num_patches=100,      # per slide
    magnification=20,
    tissue_threshold=0.6,
    sampling="random"
)

from Utils.graphbuilder import build_graphs_from_loader

graph_save_dir = "GraphDataset"

graphs = build_graphs_from_loader(
    loader_output=wsi_loader_out,
    save_dir=graph_save_dir,
    spatial_radius=512,
    sim_k=8,
    slide_label_key='label'
)

from Utils.dataset import CancerRecurrenceGraphDataset
from torch_geometric.loader import DataLoader

dataset = CancerRecurrenceGraphDataset(root=graph_save_dir, graph_type="combined")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example iteration
for batch in loader:
    print(batch)
    print(f"Number of graphs in batch: {batch.num_graphs}")
    break