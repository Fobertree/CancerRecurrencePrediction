import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity

from wsi import load_wsi # load as np arrays

# dinov2_vits14 is one specific model undero dinov2. Use this for now
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

def load_patches(path: str):
    '''
    TODO: implement this
    
    loads saved subpatches generated from wsi.py (assuming DINOv2 on the raw WSI is too expensive, we can do some run attempts on this)
    
    '''
    loaded_wsi_patches, corners = load_wsi(path, threshold=1) # load at most threshold WSI
    processed_patches = []

    # build graphs
    for single_wsi_patches in loaded_wsi_patches:
        data_spat, data_sem = patch_to_graph(single_wsi_patches, patch_corners=corners)
        
    
    # segmentation + embedding

    pass

def patch_to_graph(wsi_patches: list[np.ndarray | torch.Tensor], 
                   patch_corners: list[tuple[float,float]],
                   slide_label: int,
                   spatial_radius: float = 512, # pixel radius
                   sim_threshold: float = 0.8):
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
            # patch_feature = patch_features.mean(dim=0)
            features.append(patch_features)
    
    num_patches = patch_features.shape[0]
    x = torch.stack(features)

    # build edges
    # spatial similarity - KDTree
    # NOTE: if we want to augment the data via scaling or rotation, we need to switch to centers
    # since they are scale+rotation-invariant. Otherwise fine
    kd_tree = KDTree(patch_corners) # use upper-left patch corners from openslide
    pairs = kd_tree.query_pairs(spatial_radius)
    edge_index_spat = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    
    # feature similarity - Cosine similarity
    sim = cosine_similarity(x.numpy())
    i,j = np.where(sim > sim_threshold)
    edge_index_sem = torch.tensor([i,j], dtype=torch.long)

    # load labels
    y = torch.tensor([slide_label], dtype=torch.long) # TODO: fit labels

    # Not sure if I should make these in-memory datasets - Alex
    data_spat= Data(x=x, edge_index=edge_index_spat,y=y)
    data_sem = Data(x=x, edge_index=edge_index_sem,y=y)

    return data_spat, data_sem

if __name__ == "__main__":
    # testing
    images = load_patches("Data", threshold=1)
    img : np.ndarray = images[0]

    