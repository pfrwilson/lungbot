
from omegaconf import DictConfig, ListConfig

from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Union, Sequence
from dataclasses import dataclass

from .components.object_detection import DetectionOutput
from .components.chexnet import CheXNet
from .components.region_proposal_network import RegionProposalNetwork, RPNConfig
from .objectives.rcnn_loss import RCNNLoss
from .objectives.metric import DetectionMetric


@dataclass
class RPNModuleConfig:
    scales: Union[Sequence, ListConfig]
    aspect_ratios: Union[Sequence, ListConfig]
    freeze_chexnet: bool
    lambda_: float
    nms_iou_threshold: float
    num_training_examples_per_image: int 
    min_num_positive_examples: int
    positivity_threshold: float 
    negativity_threshold: float 
    lr: float    


class RPNModule(LightningModule):
    
    def __init__(self, config: RPNModuleConfig):

        super().__init__()

        self.config = config

        self.rpn = RegionProposalNetwork(
            RPNConfig(
                image_input_size=1024, 
                feature_map_size=32, 
                feature_dim=1024, 
                hidden_dim=256, 
                scales=config.scales, 
                aspect_ratios=config.aspect_ratios, 
                nms_threshold=config.nms_iou_threshold
            )
        )
        
        self.chexnet = CheXNet()
        if config.freeze_chexnet:
            for parameter in list(self.chexnet.parameters()):
                parameter.requires_grad = False     
        
        self.loss_fn = RCNNLoss(lambda_=config.lambda_)
    
        self.num_training_examples_per_images=config.num_training_examples_per_image
        self.min_num_positive_examples=config.min_num_positive_examples
        self.positivity_threshold=config.positivity_threshold
        self.negativity_threshold=config.negativity_threshold
        self.lr = config.lr

        self.metrics = DetectionMetric()


    def configure_optimizers(self):
        
        optim = torch.optim.Adam(self.parameters(), self.lr)
        
        scheduler = CosineAnnealingWarmRestarts(
            optim, 
            T_0=5, 
        )

        return {
            'optimizer': optim, 
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }


    def training_step(self, batch, batch_idx):
    
        loss = 0
        
        for item in batch:
            
            pixel_values, true_boxes = item
        
            feature_maps = self.chexnet(pixel_values)
            
            detection_output = self.rpn(feature_maps, training=True)
        
            training_batch = self.rpn.get_training_batch(
                detection_output, 
                true_boxes, 
                min_num_positives=self.config.min_num_positive_examples, 
                positivity_threshold=self.config.positivity_threshold, 
                negativity_threshold=self.config.negativity_threshold
            )
        
            training_batch = self.rpn.select_training_batch_subset(
                training_batch, self.config.num_training_examples_per_image
            )
        
            loss += self.loss_fn.compute_from_batch(training_batch)
            
            self.metrics.update(
                training_batch.class_scores, 
                training_batch.iou_scores_with_targets
            )
            
        metrics = self.metrics.compute()
        
        for key in metrics.keys():  
            self.log(f'train/{key}', metrics[key], prog_bar=True, logger=True)
        
        self.log('train/loss', loss, logger=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        for item in batch:
            
            pixel_values, true_boxes = item
            
            feature_maps = self.chexnet(pixel_values)
            
            detection_output: DetectionOutput = self.rpn(feature_maps)
            
            _, iou_scores = self.rpn.match_proposed_boxes_to_true(
                true_boxes, detection_output.proposed_boxes
            )
            
            self.metrics.update(
                detection_output.class_scores, 
                iou_scores
            )        
        
        metrics = self.metrics.compute()
        
        for key in metrics.keys():  
            self.log(f'val/{key}', metrics[key], prog_bar=True, logger=True)
    
    
    def test_step(self, batch, batch_idx):

        for item in batch:
            
            pixel_values, true_boxes = item
            
            feature_maps = self.chexnet(pixel_values)
            
            detection_output: DetectionOutput = self.rpn(feature_maps)
            
            _, iou_scores = self.rpn.match_proposed_boxes_to_true(
                true_boxes, detection_output.proposed_boxes
            )
            
            self.metrics.update(
                detection_output.class_scores, 
                iou_scores
            )        
        
        metrics = self.metrics.compute()
        
        for key in metrics.keys():  
            self.log(f'test/{key}', metrics[key], prog_bar=True, logger=True)
        
            
    def on_epoch_end(self) -> None:
        self.metrics.reset()

