import os
import torch
print(torch.__version__)
print(torch.version.cuda)
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import KNNGraph
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from Utils.wsi import load_wsi, load_images  # use the unified loaders
from PIL import Image
import numpy as np

# DINOv2 model
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_model.eval()  # set to eval mode

PATCH_SIZE = 224  # multiple of 14 for ViT-S/14

def extract_patch_features(patches):
    # """
    # Extract DINOv2 features for a list of numpy RGB patches.
    # Returns a tensor [num_patches, embed_dim]
    # """
    # features = []
    # with torch.no_grad():
    #     for patch in patches:
    #         if isinstance(patch, torch.Tensor):
    #             patch_tensor = patch.unsqueeze(0)  # add batch dim if needed
    #         else:
    #             patch_tensor = torch.tensor(patch).permute(2,0,1).unsqueeze(0).float() / 255.0
    #         out = dinov2_model.get_intermediate_layers(patch_tensor)[0]
    #         out = out.squeeze(0)[1:]  # remove CLS token
    #         agg = out.mean(dim=0)
    #         features.append(agg)
    # return torch.stack(features)

    """
    Extract DINOv2 features for a list of RGB patches.
    Batch processing for speed. Returns [num_patches, embed_dim]
    """
    # stack all patches into a single tensor: [B, C, H, W]
    patch_tensor = torch.stack(patches)  # already [C,H,W] from preprocessing

    with torch.no_grad():
        out = dinov2_model.get_intermediate_layers(patch_tensor)[0]  # [B, num_tokens, dim]
        out = out[:, 1:, :]             # remove CLS token
        out = out.mean(dim=1)           # average pooling over tokens

    return out

def build_spatial_graph(patch_centers, x, radius=512):
    """
    Build a spatial adjacency graph using KDTree radius query.
    """

    patch_centers = np.array(patch_centers, dtype=float)

    # Ensure correct shape
    if patch_centers.ndim == 1:
        patch_centers = patch_centers.reshape(-1, 2)
    elif patch_centers.shape[1] != 2:
        raise ValueError(f"patch_centers must have shape [num_patches, 2], got {patch_centers.shape}")

    kd = KDTree(patch_centers)
    pairs = kd.query_pairs(radius)
    if not pairs:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    return Data(x=x, pos=torch.tensor(patch_centers, dtype=torch.float), edge_index=edge_index)

def build_similarity_graph(x, k=8):
    """
    Build a k-NN graph based on cosine similarity using CPU-friendly KDTree.
    """
    x_np = x.cpu().numpy()
    # Normalize for cosine similarity
    x_norm = x_np / np.linalg.norm(x_np, axis=1, keepdims=True)
    kd = KDTree(x_norm)
    neighbors = kd.query(x_norm, k=k+1)  # include self
    indices = neighbors[1][:, 1:]         # remove self
    source, target = [], []
    for i, nbrs in enumerate(indices):
        for j in nbrs:
            source.append(i)
            target.append(j)
    edge_index = torch.tensor([source, target], dtype=torch.long)
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
            # DINOv2 requires H and W to be multiples of 14
            PATCH_SIZE = 224  # common multiple of 14

            patch_arrays = []
            for f in patch_files:
                img = Image.open(f).convert("RGB")               # ensure RGB
                img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)  # resize to multiple of 14
                patch_arrays.append(torch.tensor(np.array(img).transpose(2,0,1), dtype=torch.float)/255.0)

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

        # Determine a unique slide ID
        if "svs_name" in metadata:
            slide_name = os.path.splitext(os.path.basename(metadata["svs_name"]))[0]
        elif patch_files:
            slide_name = os.path.splitext(os.path.basename(patch_files[0]))[0]
        else:
            slide_name = f"{idx:04d}"  # fallback if no filename available

        # Save graphs with slide_name instead of idx
        torch.save(spat_graph, os.path.join(save_dir, f"spat_graph_{slide_name}.pt"))
        torch.save(sim_graph, os.path.join(save_dir, f"sim_graph_{slide_name}.pt"))
        torch.save(combined_graph, os.path.join(save_dir, f"combined_graph_{slide_name}.pt"))

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