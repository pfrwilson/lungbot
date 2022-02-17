
from omegaconf import DictConfig, ListConfig

from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataclasses import dataclass

from torchmetrics import Precision, Recall

from .components.object_detection import DetectionOutput
from .components.chexnet import CheXNet
from .components.region_proposal_network import RegionProposalNetwork
from .objectives.rcnn_loss import RCNNLoss

from ..configs.schema import RPNModuleConfig


class RPNModule(LightningModule):
    
    def __init__(self, config: RPNModuleConfig):

        super().__init__()
        
        self.config = config
        self.rpn = RegionProposalNetwork(
            config.rpn_config
        )
        
        self.chexnet = CheXNet()
        if config.freeze_chexnet:
            for parameter in list(self.chexnet.parameters()):
                parameter.requires_grad = False     
        
        self.loss_fn = RCNNLoss(lambda_=config.lambda_)

        self.train_recall = Recall()
        self.train_precision = Precision()
        self.val_recall = Recall()
        self.val_precision = Precision()
        self.test_recall = Recall()
        self.test_precision = Precision()

        self.save_hyperparameters()

    def configure_optimizers(self):
        
        optim = torch.optim.Adam(self.parameters(), self.config.lr)
        
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
            
            preds = F.softmax(training_batch.class_scores, dim=-1)[:, 1]
            labels = training_batch.class_labels
            
            self.train_precision(preds, labels)
            self.train_recall(preds, labels)
        
        self.log('train/loss', loss)
        self.log('train/precision', self.train_precision, on_step=False)
        self.log('train/recall', self.train_recall, on_step=False)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        for item in batch:
            
            pixel_values, true_boxes = item
            
            feature_maps = self.chexnet(pixel_values)
            
            detection_output: DetectionOutput = self.rpn(feature_maps)
            
            _, iou_scores = self.rpn.match_proposed_boxes_to_true(
                true_boxes, detection_output.proposed_boxes
            )
            
            preds = F.softmax(detection_output.class_scores, dim=-1)[:, 1]
            targets = (iou_scores >= self.config.metrics_match_threshold).long()
        
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
        
        self.log('val/precision', self.val_precision, on_epoch=True)
        self.log('val/recall', self.val_recall, on_epoch=True)
    
    def test_step(self, batch, batch_idx):

        for item in batch:
            
            pixel_values, true_boxes = item
            
            feature_maps = self.chexnet(pixel_values)
            
            detection_output: DetectionOutput = self.rpn(feature_maps)
            
            _, iou_scores = self.rpn.match_proposed_boxes_to_true(
                true_boxes, detection_output.proposed_boxes
            )
            
            preds = F.softmax(detection_output.class_scores, dim=-1)[:, 1]
            targets = (iou_scores >= self.config.metrics_match_threshold).long()
        
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
        
        self.log('test/precition', self.test_precision, on_epoch=True)
        self.log('test/recall', self.test_recall, on_epoch=True)
        
            


