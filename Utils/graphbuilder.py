import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

from wsi import load_images # load as np arrays

# dinov2_vits14 is one specific model undero dinov2. Use this for now
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

def load_patches(path: str):
    '''
    TODO: implement this
    
    loads saved subpatches generated from wsi.py (assuming DINOv2 on the raw WSI is too expensive, we can do some run attempts on this)
    
    '''
    wsis_patches = load_images("Data", threshold=1) # load at most threshold WSI

    # build graphs
    for wsi_patches in wsis_patches:
        for patch in wsi_patches:
            graph = patch_to_graph()

    pass

def patch_to_graph(patch: np.ndarray | torch.Tensor):
    with torch.no_grad():
        features = dinov2_model.get_intermediate_layers(patch)[0]
        patch_features = features.squeeze(0)
        # rm CLS token
        patch_features = patch_features[1:]
    
    num_patches = patch_features.shape[0]


if __name__ == "__main__":
    # testing
    images = load_images("Data", threshold=1)
    img : np.ndarray = images[0]

    