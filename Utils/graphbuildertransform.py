'''
Single graph construction by direct 8-neighbor adjacency
'''

import os
import ssl
import pandas as pd
import logging
from PIL import Image
import torch
import numpy as np
import os.path as osp
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.neighbors import KDTree # for edge construction within L2 dist threshold of patch centers
from tqdm import tqdm
import torch.nn.functional as F
import itertools

# from graphbuilder import extract_patch_features

logger = logging.Logger("graphbuild", level=logging.DEBUG)
log_file = 'Logs/graphbuild.log'
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s -%(levelname)s :: %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# DINOv2 model - this isn't getting used yet

DISABLE_SSL = True

if DISABLE_SSL:
    # WARNING: this can be a security vulnerability but this is to fix SSL issue with torchhub load DINOv2
    # I attempted this as a fix to some sort of SSL error/connection error (for some reason the requests fully fail instead of returning 400/500)
    # I assume we aren't rate-limited (if one exists) bc we only load it once per run?
    ssl._create_default_https_context = ssl._create_unverified_context

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_model.eval()  # set to eval mode

PATCH_SIZE = 28  # multiple of 14 for ViT-S/14

# COPYPASTED FROM GRAPHBUILDER.PY TO SIDESTEP UTIL PATH IMPORT CONFLICT
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


def build_graphs(patch_dir = "dinov2_patches", save_dir="GraphDataset", metadata_path = "Data/new_metadata.csv", save_graph = True, replace = False):
    '''
    Instead of relying on loader_output from previous graphbuilder, designed to be fully independent

    @replace: To skip reconstrucitng graphs that already exist (based on path), set to False

    DEPRECATED - DO NOT CALL
    '''
    os.makedirs(save_dir, exist_ok=True)
    all_graphs = []
    metadata_df = pd.read_csv(metadata_path, index_col=0).set_index('svs_name')
    
    for wsi_patch_dir in tqdm(os.listdir(patch_dir)):
        fpath = osp.join(patch_dir, wsi_patch_dir)
        if osp.isfile(fpath):
            continue

        # wsi detected (dir), build graph based on patches

        wsi_patch_dir_upper = wsi_patch_dir.upper()
        graph_save_name = osp.join(save_dir, f"graphtransformer_{wsi_patch_dir_upper}.pt")

        if not replace and osp.exists(graph_save_name):
            # skip. Graph already 
            logger.warning(f"Detected existence of graph:: {graph_save_name}. Skipping...")
            graph = torch.load(graph_save_name, weights_only=False)
            all_graphs.append(graph)
            continue

        if wsi_patch_dir_upper not in metadata_df.index:
            logger.error(f'Could not locate in DF:: {wsi_patch_dir_upper}')
        wsi_metadata = metadata_df.loc[wsi_patch_dir_upper]
        label = wsi_metadata['Oncotype DX Breast Recurrence Score']

        patch_arrays = []
        patch_centers = []
        for patch_file in os.listdir(fpath):
            # THIS IS HARDCODED. MUST MATCH FORMAT
            params = patch_file.removeprefix('patch_').removesuffix('.png').split("_")
            wsi_id, x_pos, y_pos = params
            x_pos, y_pos = int(x_pos), int(y_pos)
            patch_centers.append((x_pos, y_pos))
            # logger.info(params)

            # migrated logic from graphbuilder.py - build_graphs_from_loader
            img = Image.open(osp.join(fpath, patch_file)).convert("RGB")               # ensure RGB
            img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)  # resize to multiple of 14
            patch_arrays.append(torch.tensor(np.array(img).transpose(2,0,1), dtype=torch.float)/255.0)
        
        if len(patch_arrays) == 0:
            print(f"COULD NOT FIND PATCHES FOR {wsi_patch_dir}")
            continue

        x = extract_patch_features(patch_arrays)
        y = torch.tensor([label], dtype=torch.long)
        # print(patch_centers)
        pos = torch.tensor(patch_centers, dtype=torch.float)

        # kdtree to build edge_index
        kdtree = KDTree(pos)
        # RADIUS_THRESHOLD = 1000 # another hyperparameter >:(
        N_NEIGHBORS = 10 # KNN
        NUM_NODES = pos.shape[0]

        distances, indices = kdtree.query(pos,k=N_NEIGHBORS)
        src_list = []
        dst_list = []

        for i in range(NUM_NODES):
            # skip j = 0. First neighbor is itself
            for j in range(1, N_NEIGHBORS):
                src_list.append(i)
                dst_list.append(indices[i,j])
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Compute pairwise L2 distances for edges
        src, dst = edge_index
        edge_weight = torch.norm(pos[src] - pos[dst], dim=1)

        wsi_graph_directed = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos=pos,
            edge_attr=edge_weight.unsqueeze(1) #[E,F]
        )

        #pe_dim = 4 * num_bands. pos is dim 2, times 2 since variation of coeff. sin/cos per freq (with num_bands # freq)
        wsi_graph_directed.pe = fourier_encode(wsi_graph_directed.pos, num_bands=8)

        # make undirected
        undirected_transform = T.ToUndirected()
        wsi_graph_undirected = undirected_transform(wsi_graph_directed)

        # replaced slide_name with wsi_patch_dir_upper
        if save_graph:
            torch.save(wsi_graph_directed, graph_save_name)
            logger.info(f"Saved graph: graphtransformer_{wsi_patch_dir_upper}.pt")
        
        # Not sure if we want this or just load from dir
        # seems like we are just load from dir so will prob rm this later - Alex
        all_graphs.append(wsi_graph_undirected)
    
    return all_graphs

def build_graphs_seq(patch_dir = "dinov2_patches_seq", save_dir="GraphDatasetSeq", metadata_path = "Data/new_metadata.csv", save_graph = True, replace = True):
    os.makedirs(save_dir, exist_ok=True)
    all_graphs = []
    metadata_df = pd.read_csv(metadata_path, index_col=0).set_index('svs_name')
    
    # TODO: replace with process pool executor
    for wsi_patch_dir in tqdm(os.listdir(patch_dir)):
        fpath = osp.join(patch_dir, wsi_patch_dir)
        if osp.isfile(fpath):
            continue

        # wsi detected (dir), build graph based on patches

        wsi_patch_dir_upper = wsi_patch_dir.upper()
        graph_save_name = osp.join(save_dir, f"graphtransformer_{wsi_patch_dir_upper}.pt")

        if not replace and osp.exists(graph_save_name):
            # skip. Graph already 
            logger.warning(f"Detected existence of graph:: {graph_save_name}. Skipping...")
            graph = torch.load(graph_save_name, weights_only=False)
            all_graphs.append(graph)
            continue

        if wsi_patch_dir_upper not in metadata_df.index:
            logger.error(f'Could not locate in DF:: {wsi_patch_dir_upper}')
        wsi_metadata = metadata_df.loc[wsi_patch_dir_upper]
        label = wsi_metadata['Oncotype DX Breast Recurrence Score']

        patch_arrays = []
        patch_centers = []
        patch_center_indices = {}
        patch_idx = 0
        for patch_file in os.listdir(fpath):
            # THIS IS HARDCODED. MUST MATCH FORMAT
            params = patch_file.removeprefix('patch_').removesuffix('.png').split("_")
            wsi_id, x_pos, y_pos = params
            x_pos, y_pos = int(x_pos), int(y_pos)
            patch_centers.append((x_pos, y_pos))
            patch_center_indices[(x_pos, y_pos)] = patch_idx #indexer for edge_index construction
            patch_idx += 1
            # logger.info(params)

            # migrated logic from graphbuilder.py - build_graphs_from_loader
            img = Image.open(osp.join(fpath, patch_file)).convert("RGB")               # ensure RGB
            img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)  # resize to multiple of 14
            patch_tensor = torch.tensor(np.array(img).transpose(2,0,1), dtype=torch.float)/255.0
            patch_arrays.append(patch_tensor)
        
        if len(patch_arrays) == 0:
            print(f"COULD NOT FIND PATCHES FOR {wsi_patch_dir}")
            continue
        
        # dinov2 patch embeddings
        pe = extract_patch_features(patch_arrays)
        y = torch.tensor([label], dtype=torch.long)
        x = torch.arange(len(patch_arrays), dtype=torch.long).unsqueeze(-1)
        # print(patch_centers)
        pos = torch.tensor(patch_centers, dtype=torch.float)

        # TODO: neighbor edge_index construction by immediate 8-neighbor adjacency
        edge_index = create_8_neighbor_edge_index(patch_center_indices)
        # TODO: edge weights by cosine similarity of neighbors
        src, dst = edge_index
        patches_tensor = torch.stack(patch_arrays).double()  # shape [N, 3, H, W]
        patches_flat = patches_tensor.flatten(start_dim=1)   # shape [N, 3*H*W]

        edge_weight = F.cosine_similarity(
            patches_flat[src], patches_flat[dst], dim=1
        )

        wsi_graph_directed = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos=pos,
            pe=pe,
            edge_attr=edge_weight.unsqueeze(1) #[E,F]
        )

        print(x.shape, pe.shape, edge_index.shape, y.shape, pos.shape, edge_weight.unsqueeze(1).shape)

        # make undirected
        undirected_transform = T.ToUndirected()
        wsi_graph_undirected = undirected_transform(wsi_graph_directed)

        # replaced slide_name with wsi_patch_dir_upper
        if save_graph:
            torch.save(wsi_graph_directed, graph_save_name)
            logger.debug(f"Saved graph: graphtransformerseq_{wsi_patch_dir_upper}.pt")
        
        # Not sure if we want this or just load from dir
        # seems like we are just load from dir so will prob rm this later - Alex
        all_graphs.append(wsi_graph_undirected)

        # WE NEED CONSISTENT SHAPE IN X.dim and there pe.dim
        # TODO: create projection to project all dimensions into common dimensionality
        # find max dims across dataset
        max_x_dim = max(g.x.shape[1] for g in all_graphs)
        max_pe_dim = max(g.pe.shape[1] for g in all_graphs)

        # pad function
        def pad_tensor(t, target_dim):
            if t.shape[1] < target_dim:
                pad_amt = target_dim - t.shape[1]
                padding = torch.zeros((t.shape[0], pad_amt), device=t.device)
                t = torch.cat([t, padding], dim=1)
            return t

        # apply
        for g in all_graphs:
            g.x = pad_tensor(g.x, max_x_dim)
            g.pe = pad_tensor(g.pe, max_pe_dim) 
                
    return all_graphs


def create_8_neighbor_edge_index(node_idx: dict):
    """
    Generates the edge_index for an 8-neighbor adjacency graph on a grid.
    
    Args:
        height (int): The height of the grid (image).
        width (int): The width of the grid (image).
        
    Returns:
        torch.Tensor: The edge_index tensor of shape [2, num_edges].
    """
    # Calculate the total number of nodes (pixels)
    
    # Initialize lists to store source and target nodes of edges
    source_nodes = []
    target_nodes = []
    
    logger.info(node_idx.keys())

    for src, src_idx in node_idx.items():
        for dst, dst_idx in node_idx.items():
            if src == dst:
                continue
                
            source_nodes.append(src_idx)
            target_nodes.append(dst_idx)
                    
    # Convert lists to a PyTorch tensor with shape [2, num_edges]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    return edge_index

def fourier_encode(pos, num_bands=8):
    """
    pos: [num_nodes, 2] coordinates
    returns: [num_nodes, 2 * num_bands * 2] encoded features
    """
    freq_bands = 2.0 ** torch.linspace(0, num_bands-1, num_bands, device=pos.device)
    pts = pos.unsqueeze(-1) * freq_bands  # [N, 2, num_bands]
    sin = torch.sin(pts)
    cos = torch.cos(pts)
    return torch.cat([sin, cos], dim=-1).flatten(1)

if __name__ == "__main__":

    # build graphs directly
    # designed to be self-contained so just run this on dinov2_patches
    # build_graphs()
    build_graphs_seq(replace=False)
