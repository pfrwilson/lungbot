from matplotlib.pyplot import cla
import torch 
import einops
from einops.layers.torch import Rearrange
from torch import det, nn 
from torchvision import ops 
from torch.nn import functional as F

from dataclasses import dataclass
from typing import Sequence, Union
from omegaconf import ListConfig
from itertools import product

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .object_detection import DetectionOutput, DetectionTrainingBatch, DetectionMixin
from ...configs.schema import RPNConfig


class RegionProposalNetwork(nn.Module, DetectionMixin):

    def __init__(self, config: RPNConfig):  
        
        super().__init__()
   
        self.config = config
    
        self.anchor_boxes = self.create_anchor_boxes(
            config.image_input_size, 
            config.feature_map_size, 
            config.scales, 
            config.aspect_ratios
        )
        
        self.sliding_window = nn.Conv2d(
            in_channels=config.feature_dim, 
            out_channels=config.hidden_dim, 
            kernel_size=3, 
            padding=1
        )
    
        k = len(config.scales) * len(config.aspect_ratios)

        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_dim, 
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
                in_channels=config.hidden_dim, 
                out_channels=k * 2, 
                kernel_size=1
            ), 
            Rearrange(
                'b (k two) h w -> b h w k two', 
                k = k, 
                two = 2
            )
        )
    
    def forward(self, in_feature_map, training=False):
        
        detection_output = self.propose_boxes(in_feature_map)

        if not training: 
            detection_output = self.apply_nms(
                detection_output,
                self.config.nms_threshold,
            )
            
        return detection_output
        
    def propose_boxes(self, in_feature_map):
        
        assert in_feature_map.shape[0] == 1, 'Call on single slice of feature map at a time.'
        _, feature_dim, H, W = in_feature_map.shape
        
        sliding_window_output = self.sliding_window(in_feature_map)
        
        regression_scores = self.bbox_regressor(sliding_window_output)
        regression_scores = einops.rearrange(
            regression_scores, 
            'b h w k four -> ( b h w k ) four'
        )
        
        objectness_scores = self.classifier(sliding_window_output)
        objectness_scores = einops.rearrange(
            objectness_scores,  
            'b h w k two -> ( b h w k ) two'
        )
        
        proposed_boxes = self.boxreg_transform(regression_scores, self.anchor_boxes)
        
        return DetectionOutput(
            proposed_boxes=proposed_boxes, 
            regression_scores=regression_scores, 
            class_scores=objectness_scores, 
            anchor_boxes=self.anchor_boxes
        )    
    
    @staticmethod
    def create_anchor_boxes(input_size: int, feature_map_size: int,
                            scales: Sequence[float], aspect_ratios: Sequence[float]):
        
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
        
        return anchor_boxes.to(DEVICE)
        