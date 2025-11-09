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

PATCH_SIZE = 224  # multiple of 14 for ViT-S/14

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
            graph = torch.load(graph_save_name)
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


if __name__ == "__main__":

    # build graphs directly
    # designed to be self-contained so just run this on dinov2_patches
    build_graphs()