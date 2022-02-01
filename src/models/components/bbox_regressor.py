
import torch
from torch import nn 
from .mlp import MLP
from torch.nn import functional as F
from einops.layers.torch import Rearrange
import einops


class BBoxRegressor(nn.Module):   
    """See https://arxiv.org/pdf/1311.2524.pdf Appendix C for details"""
    
    def __init__(self, in_features, hidden_size, num_classes):
        """A module which performs bounding box regression
            based on the final feature vector of the RCNN (following
            RoI pooling and fully connected layers).
            The module learns to predict scale and translation factors that
            move the proposed bounding box closer to the true bounding
            box.

        Args:
            in_features (int): Number of features in the final feature vector
            hidden_size (int): Number of units in the hidden layers of the module
            num_classes (int): Number of object classes to be detected
        """
        
        super(BBoxRegressor, self).__init__()
        
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.mlp = MLP(input_dim=self.in_features, 
                       hidden_size=self.hidden_size, 
                       output_dim=self.hidden_size)
        
        self.to_transform_scores = nn.Sequential(
            nn.Linear(
                in_features= self.hidden_size, 
                out_features= 4 * self.num_classes
            ), 
            Rearrange(
                ' b n (n_classes rect_spec_len) -> b n n_classes rect_spec_len', 
                n_classes = self.num_classes, 
                rect_spec_len = 4
            )
        )
        
    def forward(self, roi_features, proposed_boxes=None):
        """From roi features compute translation/scaling scores. 
        
        Args:
            roi_features (torch.Tensor): The roi vectors (output of RoI pooling module)
            proposed_boxes (torch.Tensor, optional): Optionally, include the box 
            proposal regions corresponding to the ROI features.
            If provided, the module will output the transformation scores 
            together with the actual predicted boxes. Defaults to None.
        """
        
        # n roi feature vectors per image, b images per batch.
        b, n, d = roi_features.shape
        assert d == self.in_features
        
        # n rectangles per image in batch, each rectangle 
        # specified by x, y, w, h.
        if proposed_boxes is not None:
            b_, n_, rect_spec_len = proposed_boxes.shape
            assert rect_spec_len == 4, 'Size error in rectangle specification. Expected size 4.'
            assert (b == b_ and n == n_), 'Size mismatch between feature vectors and box proposals'
        
        x = roi_features
        x = self.mlp(x)
        x = F.relu(x)
        
        transform_scores = self.to_transform_scores(x)
        
        out_dict = {}
        out_dict['transform_scores'] = transform_scores
        
        if proposed_boxes is None: 
            return out_dict
        
        proposed_boxes = einops.repeat(
            proposed_boxes, 
            'b n_boxes rect_spec_len -> b n_boxes n_classes rect_spec_len', 
            b = b,
            n_boxes = n, 
            n_classes = self.num_classes,
            rect_spec_len = 4,
        )

        P_x = proposed_boxes[..., [0]]
        P_y = proposed_boxes[..., [1]]
        P_w = proposed_boxes[..., [2]]
        P_h = proposed_boxes[..., [3]]
        
        d_x = transform_scores[..., [0]]
        d_y = transform_scores[..., [1]]
        d_w = transform_scores[..., [2]]
        d_h = transform_scores[..., [3]]     
        
        G_x = P_w * d_x + P_x
        G_y = P_h * d_y + P_y
        G_w = P_w * torch.exp2(d_w)
        G_h = P_h * torch.exp2(d_h)    
        
        predicted_boxes = torch.cat(
            [G_x, G_y, G_w, G_h], dim = -1
        )
        
        out_dict['predicted_boxes'] = predicted_boxes
        out_dict['proposed_boxes'] = proposed_boxes
        
        return out_dict
    
    
        