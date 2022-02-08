
from omegaconf import DictConfig

from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
import torch

from .components.chexnet import CheXNet
from .components.region_proposal_network import RegionProposalNetwork
from .components.rcnn_loss import RCNNLoss

class RPNSystem(LightningModule):
    
    def __init__(
        self, 
        scales = [.5, 1, 2],
        aspect_ratios = [1, 1.5, .66], 
        freeze_chexnet = True, 
        lambda_ = 1.0, 
        nms_iou_threshold = 0.5, 
        num_training_examples_per_image = 16, 
        min_num_positive_examples = 4,
        lr = 1e-3
    ):

        super().__init__()

        self.rpn = RegionProposalNetwork(
            image_input_size=1024, 
            feature_map_size=32, 
            feature_dim=1024, 
            hidden_dim=256, 
            scales=scales, 
            aspect_ratios=aspect_ratios
        )
        
        self.chexnet = CheXNet()
        if freeze_chexnet:
            for parameter in list(self.chexnet.parameters()):
                parameter.requires_grad = False     
        
        self.loss_fn = RCNNLoss(lambda_=lambda_)
    
        for module in self.modules():
            module = module.double()
    
        self.nms_iou_threshold = torch.tensor(nms_iou_threshold).double()
        self.num_training_examples_per_images=num_training_examples_per_image
        self.min_num_positive_examples=min_num_positive_examples
        self.lr = lr

        self.map = MeanAveragePrecision(box_format='xywh')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
    
        pixel_values, true_boxes, true_box_indices = batch
        batch_size = pixel_values.shape[0]
            
        feature_maps = self.chexnet(pixel_values)
        
        out_dict = self.rpn.propose_boxes(feature_maps)
        
        loss = 0
        for idx in range(batch_size):
            
            training_examples = self.rpn.select_training_examples(
                out_dict['proposed_boxes'][idx], 
                out_dict['anchor_boxes'][idx],
                out_dict['regression_scores'][idx],
                out_dict['objectness_scores'][idx],
                true_boxes[true_box_indices == idx],
                num_training_examples=self.num_training_examples_per_images,
                min_num_positives=self.min_num_positive_examples, 
            )

            loss += self.loss_fn(
                training_examples['objectness_scores'], 
                training_examples['regression_scores'], 
                training_examples['target_regression_scores'], 
                training_examples['labels']
            )
            
            self.log('positives', len(training_examples['labels'][training_examples['labels'] == 1]), 
                     prog_bar=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        pixel_values, true_boxes, true_box_indices = batch
        batch_size = pixel_values.shape[0]
 
        feature_maps = self.chexnet(pixel_values)
        
        out_dict = self.rpn.propose_boxes(feature_maps)
        
        preds = []
        targets = []
        
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
            object_probs = nms_output['object_probs']
            
            preds.append({
                'boxes': proposed_boxes, 
                'scores': object_probs, 
                'labels': torch.IntTensor([0]*len(proposed_boxes))
            })
            
            targets.append({
                'boxes': true_boxes_for_item, 
                'labels': torch.IntTensor([0]*len(true_boxes_for_item))
            })
            
        map = self.map(preds, targets)['map']
        
        self.log('mean_average_precision', map, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        
        pixel_values, true_boxes, true_box_indices = batch
        batch_size = pixel_values.shape[0]
 
        feature_maps = self.chexnet(pixel_values)
        
        out_dict = self.rpn.propose_boxes(feature_maps)
        
        preds = []
        targets = []
        
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
            object_probs = nms_output['object_probs']
            
            preds.append({
                'boxes': proposed_boxes, 
                'scores': object_probs, 
                'labels': torch.IntTensor([0]*len(proposed_boxes))
            })
            
            targets.append({
                'boxes': true_boxes_for_item, 
                'labels': torch.IntTensor([0]*len(true_boxes_for_item))
            })
            
        map = self.map(preds, targets)['map']
        
        self.log('mean_average_precision', map, prog_bar=True)
        
    
            
            