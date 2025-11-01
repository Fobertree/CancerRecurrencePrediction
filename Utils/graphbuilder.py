import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import KNNGraph
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from wsi import load_wsi, load_images  # use the unified loaders

# DINOv2 model
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_model.eval()  # set to eval mode

def extract_patch_features(patches):
    """
    Extract DINOv2 features for a list of numpy RGB patches.
    Returns a tensor [num_patches, embed_dim]
    """
    features = []
    with torch.no_grad():
        for patch in patches:
            if isinstance(patch, torch.Tensor):
                patch_tensor = patch.unsqueeze(0)  # add batch dim if needed
            else:
                patch_tensor = torch.tensor(patch).permute(2,0,1).unsqueeze(0).float() / 255.0
            out = dinov2_model.get_intermediate_layers(patch_tensor)[0]
            out = out.squeeze(0)[1:]  # remove CLS token
            agg = out.mean(dim=0)
            features.append(agg)
    return torch.stack(features)

def build_spatial_graph(patch_centers, x, radius=512):
    """
    Build a spatial adjacency graph using KDTree radius query.
    """
    kd = KDTree(patch_centers)
    pairs = kd.query_pairs(radius)
    if not pairs:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    return Data(x=x, pos=torch.tensor(patch_centers, dtype=torch.float), edge_index=edge_index)

def build_similarity_graph(x, k=8):
    """
    Build a k-NN graph based on cosine similarity of patch features.
    """
    edge_index = knn_graph(x, k=k, loop=False, flow='target_to_source', cosine=True)
    return Data(x=x, edge_index=edge_index)

def build_graphs_from_loader(loader_output, save_dir="GraphDataset", spatial_radius=512, sim_k=8, slide_label_key='label'):
    """
    Takes output from load_wsi or load_images and builds spatial + similarity graphs.
    Saves graphs in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_graphs = []

    for idx, item in enumerate(loader_output):
        if len(item) == 4:
            # load_images: patch_files, patch_centers, patch_arrays, metadata
            patch_files, patch_centers, patch_arrays, metadata = item
        else:
            # load_wsi: patch_files, patch_centers, metadata
            patch_files, patch_centers, metadata = item
            patch_arrays = [torch.tensor(Image.open(f)) for f in patch_files]

        x = extract_patch_features(patch_arrays)
        y = torch.tensor([metadata.get(slide_label_key, -1)], dtype=torch.long)

        # build graphs
        spat_graph = build_spatial_graph(patch_centers, x, radius=spatial_radius)
        spat_graph.y = y

        sim_graph = build_similarity_graph(x, k=sim_k)
        sim_graph.y = y

        combined_graph = Data(
            x=x,
            edge_index=torch.cat([spat_graph.edge_index, sim_graph.edge_index], dim=1),
            y=y,
            pos=torch.tensor(patch_centers, dtype=torch.float)
        )

        # save
        torch.save(spat_graph, os.path.join(save_dir, f"spat_graph_{idx}.pt"))
        torch.save(sim_graph, os.path.join(save_dir, f"sim_graph_{idx}.pt"))
        torch.save(combined_graph, os.path.join(save_dir, f"combined_graph_{idx}.pt"))

        all_graphs.append((spat_graph, sim_graph, combined_graph))

    return all_graphs

if __name__ == "__main__":
    # Example usage with preprocessed JPEGs
    loader_out = load_images(
        directory_path="dinov2_jpegs",
        metadata_path="Data/Metadata.csv",
        output_dir="dinov2_patches_from_jpeg",
        patch_size=256,
        num_patches=100,
        tissue_threshold=0.6,
        return_images=True
    )

    graphs = build_graphs_from_loader(loader_out)
    print(f"Generated {len(graphs)} WSI graphs")