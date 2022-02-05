from re import S
import einops
from pytorch_lightning import LightningModule
from torchvision.ops import RoIPool, box_convert
from torch import nn
import torch

from .components import CheXNet, MLP, BBoxRegressor
from ..data.processing.roi_sampling import ROISamplerBase
from .objectives.loss_functions import RCNNLoss

class LungBot(LightningModule):
    
    def __init__(self, config):
    
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)
                
        self.num_classes = config.num_classes
        self.lambda_ = config.lambda_
                
        self.chexnet = CheXNet().densenet121.features
        if self.config.freeze_chexnet:
            for parameter in list(self.chexnet.parameters()):
                parameter.requires_grad = False     
        
        self.roi_pool = RoIPool(
            output_size=1, 
            spatial_scale=32/1024
        )
        self.roi_feature_extractor = nn.Sequential(
            nn.Flatten(start_dim=2),
            MLP(1024, 512, 256), 
            MLP(256, 128, 64)
        )
        self.bbox_regressor = BBoxRegressor(64, 32, 2)
        self.classifier = MLP(64, 32, 2)
        
        self.rcnn_loss = RCNNLoss(self.lambda_)
        
        for module in self.modules():
            module = module.double()
        
    def forward(self, img, proposed_boxes):
        
        b, c, h, w = img.shape
        
        feature_maps = self.chexnet(img)
        assert feature_maps.shape == (b, 1024, 32, 32)
        
        proposed_boxes_xyxy = box_convert(proposed_boxes, 'xyxy', 'xywh')
        roi_features = self.roi_pool(
            feature_maps, 
            [proposed_boxes_xyxy[i, ...] for i in range(proposed_boxes_xyxy.shape[0])]
        )
        # roi pool folds proposals into batch dimension. 
        # we must unfold them back:
        roi_features = einops.rearrange(
            roi_features, '(b n) c h w -> b n c h w', 
            b=b
        )
        
        roi_features = self.roi_feature_extractor(roi_features)

        label_logits = self.classifier(roi_features)
        predicted_boxes = self.bbox_regressor(roi_features, proposed_boxes)['predicted_boxes']
                
        return label_logits, predicted_boxes
        
    def training_step(self, batch, batch_idx):
        
        img, true_boxes, proposed_boxes = batch 
        
        label_logits, predicted_boxes = self(img, proposed_boxes)
        
        # fold proposal number and batch number dimensions together
        true_boxes = einops.rearrange(
            true_boxes, 'b n five -> (b n) five'
        )
        label_logits = einops.rearrange(
            label_logits, 'b n num_classes -> ( b n ) num_classes ', 
        )
        predicted_boxes = einops.rearrange(
            predicted_boxes, 
            'b n num_classes four -> (b n) num_classes four',
            four = 4
        )
        
        loss = self.rcnn_loss(label_logits, predicted_boxes, true_boxes)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
        