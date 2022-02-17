

import einops
from torch import Tensor, nn
import torch
from torch.nn import functional as F

from ..components.object_detection import DetectionTrainingBatch

def classification_loss(class_scores, labels):
    return F.cross_entropy(class_scores, labels.long())


def smooth_l1_norm(tensor, dim=-1):
    """Calculates the smooth l1 norm along the specified dimension of the tensor"""
    x = tensor
    x = torch.where(
        torch.abs(x) < 1, 
        .5 * x**2,
        torch.abs(x) - 0.5
    )
    
    return torch.sum(x, dim=dim)


def box_regression_loss(regression_scores, target_regression_scores, labels):
    
    # select non-background indices 
    non_background_indices = labels != 0
    
    regression_scores = regression_scores[non_background_indices]
    target_regression_scores = target_regression_scores[non_background_indices]
    
    norms = smooth_l1_norm(regression_scores - target_regression_scores)
    
    # get average l1 norm 
    return einops.reduce( norms, 'n -> ()', 'mean')
        
        
class RCNNLoss(nn.Module):
    
    def __init__(self, lambda_=1):
        
        super().__init__()
        
        self.lambda_ = lambda_

    def forward(self, class_scores, regression_scores, 
                target_regression_scores, labels):
        
        cls_loss = classification_loss(class_scores, labels)
        loc_loss = box_regression_loss(regression_scores, target_regression_scores, labels)    
        
        return cls_loss + self.lambda_ * loc_loss
    
    def compute_from_batch(self, batch: DetectionTrainingBatch):
        
        class_scores = batch.class_scores
        regression_scores = batch.regression_scores
        target_regression_scores = batch.target_regression_scores
        labels = batch.class_labels
        
        return self(class_scores, regression_scores, target_regression_scores, labels)