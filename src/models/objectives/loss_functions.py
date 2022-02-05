
import einops
from torch import Tensor, nn
import torch
from torch.nn import functional as F

class RCNNLoss(nn.Module):
    """ multi-task loss as described in https://arxiv.org/pdf/1504.08083.pdf """
    
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, label_logits, predicted_boxes, true_boxes,):
        """Compute the sum of class accuracy loss and bounding box regression loss.

        Args:
            label_logits (Tensor): shape '(batch_size * num_box_predictions), num_classes'
            predicted boxes (Tenosor): A tensor of shape (batch_size * num_box_predictions), num_classes, 
                4, representing the predicted true bounding box for each class for each data point.
            true_boxes (Tensor): shape '(batch_size * num_box_predictions), 5'
                a tensor containing (x, y, w, h, label) of the true bounding boxes
        """
        
        true_labels = true_boxes[..., 4].long()
        
        class_accuracy_loss = F.cross_entropy(label_logits, true_labels)
        
        true_boxes_where_not_background = []
        predicted_boxes_for_true_class = []
        
        for i in range(true_boxes.shape[0]):
            if true_labels[i] != 0:
                true_boxes_where_not_background.append(
                    true_boxes[i, :4]
                )
                predicted_boxes_for_true_class.append(
                    predicted_boxes[i, true_labels[i], :]    
                )
                
        true_boxes_where_not_background = torch.stack(
            true_boxes_where_not_background, dim=0
        )
        predicted_boxes_for_true_class = torch.stack(
            predicted_boxes_for_true_class, dim=0
        )
        
        loc_loss_for_examples = self.smooth_l1(
            true_boxes_where_not_background - predicted_boxes_for_true_class
        )
        
        loc_loss = einops.reduce(
            loc_loss_for_examples, ' b xywh -> b', 'sum'
        )
        
        loc_loss = einops.reduce(
            loc_loss, 'b -> ()', 'mean'
        )
        
        return class_accuracy_loss + self.lambda_ * loc_loss
    
    @staticmethod      
    def smooth_l1(x: Tensor):
        
        x = torch.where(
            torch.abs(x) < 1, 
            .5 * x**2,
            torch.abs(x) - 0.5
        )

        return x
    
    
