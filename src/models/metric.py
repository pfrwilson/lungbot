
from typing import overload
import torchmetrics 
from torchmetrics import Metric
import torch
from abc import ABC, abstractmethod
from torch.nn import functional as F


from .components.object_detection_utils import (
    match_proposed_boxes_to_true,
    label_by_iou_threshold,
)

class DetectionMetric(Metric, ABC):
    
    def __init__(
        self, 
        prob_threshold_for_detection=0.5, 
        iou_threshold_for_detection=0.5, 
    ):
        super().__init__()
        
        self.prob_threshold_for_detection=prob_threshold_for_detection
        self.iou_threshold_for_detection=iou_threshold_for_detection
        
        self.add_state('true_positives', default=torch.tensor(0))
        self.add_state('true_negatives', default=torch.tensor(0))
        self.add_state('false_positives', default=torch.tensor(0))
        self.add_state('false_negatives', default=torch.tensor(0))
    
    
    def update(
        self, 
        objectness_scores,
        iou_scores = None,
        proposed_boxes = None, 
        true_boxes = None
    ):
        
        object_probabilities = F.softmax(objectness_scores, dim=-1)[:, 1]
    
        if iou_scores is None: 
            iou_scores = match_proposed_boxes_to_true(
                true_boxes, proposed_boxes,
            )['iou_scores']
        
        true_labels = \
            (iou_scores >= self.iou_threshold_for_detection).long()
        predicted_labels = \
            (object_probabilities >= self.iou_threshold_for_detection).long()
        
        true_positives = torch.logical_and(
            true_labels == 1, 
            predicted_labels == 1
        )
        
        true_negatives = torch.logical_and(
            true_labels == 0, 
            predicted_labels == 0, 
        )
        
        false_positives = torch.logical_and(
            true_labels == 0, 
            predicted_labels == 1
        )
        
        false_negatives = torch.logical_and(
            true_labels == 1, 
            predicted_labels == 0
        )
    
        self.true_positives += torch.sum(true_positives)
        self.false_positives += torch.sum(false_positives)
        self.true_negatives += torch.sum(true_negatives)
        self.false_negatives += torch.sum(false_negatives)
        
        
    def compute(self):
        
        all_metrics = dict(
            true_positives = self.true_positives, 
            true_negatives = self.true_negatives, 
            false_positives = self.false_positives, 
            false_negatives = self.false_negatives, 
            precision = self.true_positives / (self.true_positives + self.false_positives), 
            recall = self.true_positives / (self.true_positives + self.false_negatives), 
            total_detections = self.true_positives + self.false_positives, 
            total_true_lables = self.true_positives + self.false_negatives,
            total_proposals = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        
        return all_metrics

    