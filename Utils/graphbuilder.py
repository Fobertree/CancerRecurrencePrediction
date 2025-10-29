import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import KNNGraph

from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity

import os

from wsi import load_wsi # load as np arrays

# dinov2_vits14 is one specific model undero dinov2. Use this for now
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov3_model = torch.hub.load("dinov3")

def load_patches(path: str):
    '''
    TODO: implement this
    
    loads saved subpatches generated from wsi.py (assuming DINOv2 on the raw WSI is too expensive, we can do some run attempts on this)
    
    '''
    # TODO: have load_wsi return labels as well @thomas
    loaded_wsi_patches, centers = load_wsi(path, threshold=1) # load at most threshold WSI

    # build graphs
    for single_wsi_patches in loaded_wsi_patches:
        data_spat, data_sem = patch_to_graph(single_wsi_patches, patch_centers=centers)
        # TODO: separate above function call, deprecate in favor of v2 rewrites
        
    
    # segmentation + embedding

    pass

def patch_to_graph(wsi_patches: list[np.ndarray | torch.Tensor], 
                   patch_centers: list[tuple[float,float]],
                   slide_label: int,
                   spatial_radius: float = 512, # pixel radius
                   sim_threshold: float = 0.8,
                   save_data = True):
    features = []
    dinov2_model.eval() # evaluation mode

    with torch.no_grad():
        # alternative: ResNet. DINOv2 is ViT-based
        for patch in wsi_patches:
            patch_features = dinov2_model.get_intermediate_layers(patch)[0]
            patch_features = patch_features.squeeze(0)
            # rm CLS token
            patch_features = patch_features[1:]
            # NOTE: not sure if should collapse this into an aggregated patch feature
            agg_patch_feature = patch_features.mean(dim=0)
            features.append(agg_patch_feature)
    
    num_patches = patch_features.shape[0]
    # [num_patches, embed_dim]
    x = torch.stack(features)

    # build edges
    # spatial similarity - KDTree
    # NOTE: if we want to augment the data via scaling or rotation, we need to switch to centers
    # since they are scale+rotation-invariant. Otherwise fine
    kd_tree = KDTree(patch_centers) # use upper-left patch corners from openslide
    pairs = kd_tree.query_pairs(spatial_radius)
    edge_index_spat = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    
    # feature similarity - Cosine similarity
    K = 8
    sim = cosine_similarity(x.numpy())
    np.fill_diagonal(sim, -np.inf)  # exclude self-similarity

    topk_indices = np.argsort(sim, axis=1)[:, -K:]
    rows = np.repeat(np.arange(sim.shape[0]), K)
    cols = topk_indices.flatten()

    edge_index_sem = torch.tensor([rows, cols], dtype=torch.long)

    edge_index = knn_graph(x, k=K, loop=False, flow='target_to_source', cosine=True)

    # load labels
    y = torch.tensor([slide_label], dtype=torch.long) # TODO: fit labels

    # Not sure if I should make these in-memory datasets - Alex
    data_spat= Data(x=x, edge_index=edge_index_spat,y=y)
    data_sem = Data(x=x, edge_index=edge_index_sem,y=y)

    combined_x = torch.concat([data_spat, data_sem], dim=0)
    combined_x_edge_index = torch.concat([edge_index_spat, edge_index_sem], dim=0)
    combined_data = Data(x= combined_x, edge_index=combined_x_edge_index,y=y)

    if save_data:
        os.mkdir("GraphDataset", exist=True)
        torch.save(data_spat, os.path.join("GraphDataset", "data_spat.pt"))
        torch.save(data_sem, os.path.join("GraphDataset", "data_sem.pt"))
        torch.save(combined_data, os.path.join("GraphDataset", "combined_data.pt"))

    return data_spat, data_sem

'''
Rewrites v2
'''
def preprocess_patches_for_graphs(wsi_patches: list[np.ndarray | torch.Tensor], 
                   patch_centers: list[tuple[float,float]],
                   slide_labels: torch.Tensor, 
                   spatial_k,       # KNN
                   sim_k,           # KNN
                   save_data = True):
    features = []
    dinov2_model.eval() # evaluation mode

    with torch.no_grad():
        # alternative: ResNet. DINOv2 is ViT-based
        for patch in wsi_patches:
            patch_features = dinov2_model.get_intermediate_layers(patch)[0]
            patch_features = patch_features.squeeze(0)
            # rm CLS token
            patch_features = patch_features[1:]
            # NOTE: not sure if should collapse this into an aggregated patch feature
            agg_patch_feature = patch_features.mean(dim=0)
            features.append(agg_patch_feature)
    
    # num_patches = patch_features.shape[0]
    # [num_patches, embed_dim]
    x = torch.stack(features)

    return x


def build_spatial_graph(
                   x,
                   patch_centers: list[tuple[float,float]],
                   slide_labels: list[int],
                   spatial_radius: float = 512, # pixel radius
                   ):
    '''
    Takes patches from one WSI only for now
    '''
    y = torch.tensor([slide_labels], dtype=torch.long)
    kd_tree = KDTree(patch_centers) # use upper-left patch corners from openslide
    pairs = kd_tree.query_pairs(spatial_radius)
    edge_index_spat = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    data_spat= Data(x=x, edge_index=edge_index_spat,y=y)
    return data_spat


# cur graph building
def build_sim_graph(
        x, #patches
        y, # labels
        k=5 # knn
):
    edge_index = knn_graph(x, k=k, loop=False, flow='target_to_source', cosine=True)
    data = Data(x=x,edge_index=edge_index, y=y)

    return data

def build_spat_graph(
        patch_centers : torch.tensor, # patch centers [num_patches, 2]
        x: torch.Tensor,              # [num_patches, feature_dim]
        y, #patch labels
        k=4
):
    data = Data(x=x, pos=patch_centers, y=y)
    transform = KNNGraph(k=k, loop=False)
    # transform should build out graph edges
    data = transform(data)
    return data

if __name__ == "__main__":
    # testing
    images = load_patches("Data", threshold=1)
    img : np.ndarray = images[0]

    