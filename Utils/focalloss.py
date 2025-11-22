# Modified from: https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/15

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Implementation of the Focal loss function

        Args:
            weight: class weight vector to be used in case of class imbalance
            gamma: hyper-parameter for the focal loss scaling.
    """
    def __init__(self, pos_weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.pos_weight = pos_weight

    def forward(self, outputs, targets):
        # ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.weight) 
        ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none", pos_weight=self.pos_weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss