import torch 
import einops
from einops.layers.torch import Rearrange
from torch import nn 
from torchvision import ops 
from torch.nn import functional as F

from typing import List
from itertools import product

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .object_detection_utils import (
    boxreg_transform, 
    inverse_boxreg_transform, 
    select_training_examples
)

class RegionProposalNetwork(nn.Module):

    def __init__(
        self, 
        image_input_size: int = 1024,
        feature_map_size: int = 32, 
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        scales: List[float] = None,
        aspect_ratios: List[float] = None, 
        nms_threshold: float = 0.7
    ):  
        super().__init__()
        
        self.image_input_size = image_input_size
        self.feature_map_size = feature_map_size
        self.feature_dim = feature_dim
        self.scales = scales
        self.aspect_ratios = aspect_ratios
    
        self.anchor_boxes = self.create_anchor_boxes(
            image_input_size, 
            feature_map_size, 
            scales, 
            aspect_ratios
        )
        
        self.sliding_window = nn.Conv2d(
            in_channels=feature_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
    
        k = len(scales) * len(aspect_ratios)

        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim, 
                out_channels=k * 4, 
                kernel_size=1
            ), 
            Rearrange(
                'b (k four) h w -> b h w k four', 
                k = k, 
                four = 4
            )
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim, 
                out_channels=k * 2, 
                kernel_size=1
            ), 
            Rearrange(
                'b (k two) h w -> b h w k two', 
                k = k, 
                two = 2
            )
        )
    
        self.boxreg_transform = boxreg_transform
        self.inverse_boxreg_transfrom = inverse_boxreg_transform
        self.select_training_examples = select_training_examples
        
        
    def propose_boxes(self, in_feature_map):
        
        b, feature_dim, H, W = in_feature_map.shape
        
        sliding_window_output = self.sliding_window(in_feature_map)
        
        regression_scores = self.bbox_regressor(sliding_window_output)
        objectness_scores = self.classifier(sliding_window_output)
        
        # expand anchor boxes into batch dimension
        anchor_boxes = einops.repeat(
            self.anchor_boxes,
            ' n_boxes four -> b n_boxes four ', 
            b=b
        )
        
        # have to fold the outer dimensions together to apply the transform:
        b, h, w, k, four = regression_scores.shape
        
        regression_scores = einops.rearrange(
            regression_scores, 
            'b h w k four -> ( b h w k ) four'
        )
        
        anchor_boxes = einops.rearrange(
            anchor_boxes,  
            'b n_boxes four -> ( b n_boxes ) four'
        )
        
        objectness_scores = einops.rearrange(
            objectness_scores,  
            'b h w k two -> ( b h w k ) two'
        )
        
        proposed_boxes = boxreg_transform(regression_scores, anchor_boxes)
        
        proposed_boxes = einops.rearrange(
            proposed_boxes, 
            '( b h w k ) four -> b ( h w k ) four', 
            b=b, h=h, w=w, k=k, four=4
        )
        
        regression_scores = einops.rearrange(
            regression_scores, 
            '( b h w k ) four -> b ( h w k ) four', 
            b=b, h=h, w=w, k=k, four=4
        )
        
        objectness_scores = einops.rearrange(
            objectness_scores, 
            ' ( b h w k ) two -> b ( h w k ) two', 
            b=b, h=h, w=w, k=k, two=2
        )
        
        anchor_boxes = einops.rearrange(
            anchor_boxes, 
            ' ( b h w k ) four -> b ( h w k ) four', 
            b=b, h=h, w=w, k=k, four=4
        )
        
        #proposed_boxes = proposed_boxes.double()
        
        return {
            'proposed_boxes': proposed_boxes, 
            'regression_scores': regression_scores,
            'objectness_scores': objectness_scores, 
            'anchor_boxes': anchor_boxes
        } 
        
        
    @staticmethod
    def apply_nms_to_region_proposals(
        proposed_boxes: torch.Tensor, 
        objectness_scores: torch.Tensor,
        iou_threshold: float,
    ):
        """applies nms to remove overlapping boxes with a lower objectness score

        Args:
            proposes_boxes (torch.Tensor): (N, 4) tensor of boxes in format xywh
            objectness_scores (torch.Tensor): (N, 2) tensor of objectness scores
            iou_threshold: The iou threshold for proposed boxes to be considered overlapping. 
            
        Returns: 
            a dict containing the proposed boxes, the objectness scores, and the object probability
            after nms. 
        """
        
        object_prob = F.softmax(objectness_scores, dim=-1)[:, 1]
        
        indices_to_keep = ops.nms(
            ops.box_convert(proposed_boxes, in_fmt='xywh', out_fmt='xyxy'),
            object_prob,
            iou_threshold=iou_threshold
        )
        
        return {
            'proposed_boxes': proposed_boxes[indices_to_keep],
            'objectness_scores': objectness_scores[indices_to_keep],
            'object_probs': object_prob[indices_to_keep]
        }
        

    @staticmethod
    def apply_objectness_threshold(proposed_boxes, object_prob, threshold):
        """Applies an threshold of objectness probability to reject
           proposed boxes with insufficient likelihodd of containing an 
           object

        Args:
            proposed_boxes ([type]): [description]
            object_prob ([type]): [description]
            threshold ([type]): [description]

        Returns:
            [type]: [description]
        """
        idx = object_prob >= threshold
        
        return {
            'proposed_boxes': proposed_boxes[idx],
            'object_prob': object_prob[idx]
        }
        
    
    @staticmethod
    def create_anchor_boxes(input_size: int, feature_map_size: int,
                            scales: List[float], aspect_ratios: List[float]):
        
        d = input_size / feature_map_size

        # CREATE THE BASE IMAGE
        def base_image_(i, j):
            
            x = i * d
            y = j * d
            w = d 
            h = d 

            return x, y, h, w
        
        indices = product(range(feature_map_size), range(feature_map_size))

        base_boxes = torch.zeros((32, 32, 4))

        for i, j in indices:
            base_boxes[i, j, :] = torch.tensor(base_image_(i, j))
            
        # CONVERT TO CXCYWH
        base_boxes_converted = einops.rearrange(
            base_boxes, 'h w l -> ( h w ) l'
        )
        base_boxes_converted = ops.box_convert(
            base_boxes_converted, 'xywh', 'cxcywh'
        )
        base_boxes_converted = einops.rearrange(
            base_boxes_converted, '( h w ) l -> h w l', 
            h = feature_map_size, w = feature_map_size
        )

        # CREATE THE ANCHOR BOXES FROM THE BASE
        def get_anchor_box_from_base(base_boxes, scale, aspect_ratio):
        
            x, y, w, h = base_boxes[0, 0, :]

            w_new = int( ( aspect_ratio **.5 ) * scale * w )
            h_new =  int( scale * w / ( aspect_ratio ** .5 ) )

            anchor_boxes = torch.zeros_like(base_boxes)
            
            indices = product(
                range(anchor_boxes.shape[0]), 
                range(anchor_boxes.shape[1])
            )
            
            for i, j in indices: 
                
                x, y, _, _ = base_boxes[i, j, :]
                anchor_boxes[i, j, :] = torch.tensor([x, y, w_new, h_new])

            return anchor_boxes
        
        anchor_boxes = []

        for scale, aspect_ratio in product(scales, aspect_ratios):
            
            anchors = get_anchor_box_from_base(base_boxes_converted, scale, aspect_ratio)
            
            anchors = einops.repeat(
                anchors, 
                'n_features_1 n_features_2 four -> n_features_1 n_features_2 1 four', 
                n_features_1 = feature_map_size, 
                n_features_2 = feature_map_size, 
                four = 4, 
            )
            
            anchor_boxes.append(anchors)
            
        anchor_boxes = torch.concat(anchor_boxes, dim = 2)    
        
        # CLIP TO THE BOUNDS OF THE INPUT IMAGE - 
        # THIS REQUIRES A FEW FORMAT CONVERSIONS
        
        n1, n2, k, four = anchor_boxes.shape
        
        anchor_boxes = einops.rearrange(
            anchor_boxes, 
            'n1 n2 k four -> ( n1 n2 k ) four'
        )

        anchor_boxes = ops.box_convert( anchor_boxes, 'cxcywh', 'xyxy')

        anchor_boxes = einops.rearrange(
            anchor_boxes,
            '( n1 n2 k ) four -> n1 n2 k four', 
            n1=n1, n2=n2, k=k, four=four
        )

        anchor_boxes = ops.clip_boxes_to_image(anchor_boxes, (input_size, input_size))

        anchor_boxes = einops.rearrange(
            anchor_boxes, 
            'n1 n2 k four -> ( n1 n2 k ) four'
        )

        anchor_boxes = ops.box_convert( anchor_boxes, 'xyxy', 'xywh')

        #anchor_boxes = einops.rearrange(
        #    anchor_boxes,
        #    '( n1 n2 k ) four -> n1 n2 k four', 
        #    n1=n1, n2=n2, k=k, four=four
        #)
        
        return anchor_boxes.to(DEVICE)
        