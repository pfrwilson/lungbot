
import torch

from src.models.objectives.loss_functions import RCNNLoss


def test_loss():
    
    b = 2
    n_boxes = 64 
    n_classes = 2
    
    true_box_spec = torch.rand( ( b * n_boxes, 4))
    true_labels = torch.randint(low=0, high=n_classes, size=((2 * 64, 1)))
    
    true_boxes = torch.concat([true_box_spec, true_labels], axis=1)
    
    pred_boxes = torch.rand( ( b * n_boxes, 2, 4) )
    
    label_logits = torch.randn( (b * n_boxes, 2 ) ) 
    
    loss_fn = RCNNLoss(lambda_= 1)
    
    loss_fn(label_logits, pred_boxes, true_boxes)    
    
