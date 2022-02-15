
from omegaconf import DictConfig, ListConfig

from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pydantic import BaseModel
from typing import Union, Sequence
from dataclasses import dataclass

from .components.chexnet import CheXNet
from .components.region_proposal_network import RegionProposalNetwork, RPNConfig
from .components.rcnn_loss import RCNNLoss

from .metric import DetectionMetric


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
    
    def __init__(
        self, 
        config: RPNModuleConfig
    ):

        super().__init__()

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
    
        self.nms_iou_threshold = torch.tensor(config.nms_iou_threshold).double()
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
    
        pixel_values, true_boxes, true_box_indices = batch
        batch_size = pixel_values.shape[0]
            
        feature_maps = self.chexnet(pixel_values)
        
        out_dict = self.rpn.propose_boxes(feature_maps)
        
        loss = 0
        
        for idx in range(batch_size):
            
            training_examples = self.rpn.select_training_examples(
                out_dict['proposed_boxes'][idx], 
                self.rpn.anchor_boxes,
                out_dict['regression_scores'][idx],
                out_dict['objectness_scores'][idx],
                true_boxes[true_box_indices == idx],
                num_training_examples=self.num_training_examples_per_images,
                min_num_positives=self.min_num_positive_examples, 
                positivity_threshold=self.positivity_threshold, 
                negativity_threshold=self.negativity_threshold, 
            )

            loss += self.loss_fn(
                training_examples['objectness_scores'], 
                training_examples['regression_scores'], 
                training_examples['target_regression_scores'], 
                training_examples['labels']
            )
            
            self.metrics.update(
                training_examples['objectness_scores'],
                training_examples['iou_scores'], 
            )
            
        metrics = self.metrics.compute()
        
        for key in metrics.keys():  
            self.log(f'train/{key}', metrics[key], prog_bar=True, logger=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        pixel_values, true_boxes, true_box_indices = batch
        batch_size = pixel_values.shape[0]
 
        feature_maps = self.chexnet(pixel_values)
        
        out_dict = self.rpn.propose_boxes(feature_maps)
        
        for idx in range(batch_size):
            
            true_boxes_for_item = true_boxes[true_box_indices == idx]
            
            proposed_boxes = out_dict['proposed_boxes'][idx]
            objectness_scores = out_dict['objectness_scores'][idx]
            
            nms_output = self.rpn.apply_nms_to_region_proposals(
                proposed_boxes, 
                objectness_scores,
                iou_threshold=self.nms_iou_threshold
            )
            
            proposed_boxes = nms_output['proposed_boxes']
            objectness_scores = nms_output['objectness_scores']
            
            self.metrics.update(
                objectness_scores,
                proposed_boxes=proposed_boxes,
                true_boxes=true_boxes_for_item
            )
        
        metrics = self.metrics.compute()
        
        for key in metrics.keys():  
            self.log(f'val/{key}', metrics[key], prog_bar=True, logger=True)
            
    def on_epoch_end(self) -> None:
        self.metrics.reset()
            
    #def test_step(self, batch, batch_idx):
    #    
    #    pixel_values, true_boxes, true_box_indices = batch
    #    batch_size = pixel_values.shape[0]
 #
    #    feature_maps = self.chexnet(pixel_values)
    #    
    #    out_dict = self.rpn.propose_boxes(feature_maps)
    #    
    #    preds = []
    #    targets = []
    #    
    #    for idx in range(batch_size):
    #        
    #        true_boxes_for_item = true_boxes[true_box_indices == idx]
    #        
    #        proposed_boxes = out_dict['proposed_boxes'][idx]
    #        objectness_scores = out_dict['objectness_scores'][idx]
    #        
    #        nms_output = self.rpn.apply_nms_to_region_proposals(
    #            proposed_boxes, 
    #            objectness_scores,
    #            iou_threshold=self.nms_iou_threshold
    #        )
    #        
    #        proposed_boxes = nms_output['proposed_boxes']
    #        object_probs = nms_output['object_probs']
    #        
    #        preds.append({
    #            'boxes': proposed_boxes, 
    #            'scores': object_probs, 
    #            'labels': torch.IntTensor([0]*len(proposed_boxes))
    #        })
    #        
    #        targets.append({
    #            'boxes': true_boxes_for_item, 
    #            'labels': torch.IntTensor([0]*len(true_boxes_for_item))
    #        })
    #        
    #    map = self.map(preds, targets)['map']
    #    
    #    self.log('mean_average_precision', map, prog_bar=True)
    #    
    #
    #        
    #        