import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

from scipy.spatial import KDTree
import torchvision.models as models
import torchvision.transforms as T
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
    for wsi_patches in loaded_wsi_patches:
        patch_to_graph(wsi_patches, corners=corners)
        
    
    # segmentation + embedding

    pass

def patch_to_graph(wsi_patches: list[np.ndarray | torch.Tensor], corners):
    features = []
    with torch.no_grad():
        # alternative: ResNet. DINOv2 is ViT-based
        for patch in wsi_patches:
            patch_features = dinov2_model.get_intermediate_layers(patch)[0]
            patch_features = patch_features.squeeze(0)
            # rm CLS token
            patch_features = patch_features[1:]
            features.append(patch_features)
    
    num_patches = patch_features.shape[0]
    x = torch.stack(features)

    # build edges
    # spatial similarity - KDTree
    kd_tree = KDTree(corners) # use upper-left patch corners from openslide
    r = 512 # pixel parameter
    pairs = kd_tree.query_pairs(r)
    edge_index_spat = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    
    # feature similarity - Cosine similarity
    sim = cosine_similarity(x.numpy())
    threshold = 0.8
    i,j = np.where(sim > threshold)
    edge_index_sem = torch.tensor([i,j], dtype=torch.long)

    # load labels
    y = torch.tensor(["PLACEHOLDER"] * num_patches) # TODO: placeholder for actual labels

    data_sem = Data(x=x, edge_index=edge_index_sem,y=y)


if __name__ == "__main__":
    # testing
    images = load_patches("Data", threshold=1)
    img : np.ndarray = images[0]

    