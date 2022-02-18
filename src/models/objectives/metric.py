
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric
import torch
from abc import ABC
from torch.nn import functional as F
from typing import Dict

from torchmetrics import AUROC, Precision, Recall


class DetectionMetricBase(Metric, ABC):
    
    def __init__(self):
        
        super().__init__()
        
        self.add_state('preds', default=torch.tensor([]))
        self.add_state('targets', default=torch.tensor([]))
        
    def update(self, preds, targets):
        
        self.preds = torch.concat([self.preds, preds], dim=0)
        self.targets = torch.concat([self.targets, targets], dim=0)


class PrecisionFROC(DetectionMetricBase):
    
    def __init__(self, fpr=None): 
        
        super().__init__()
        
        self.false_positive_rate = fpr
    
    def compute(self):
        
        thresholds = torch.arange(0, 1, 0.01)
        false_positive_rates_by_threshold = []
        precision_rates_by_threshold = []
        
        for threshold in thresholds:
            
            preds = (self.preds >= threshold).long()
            targets = self.targets
            
            false_positives = torch.sum(
                torch.logical_and(
                    preds == 1, 
                    targets == 0
                )
            )
            
            true_positives = torch.sum(
                torch.logical_and(
                    preds == 1, 
                    targets == 1
                )
            )
            
            total_negatives = torch.sum(
                targets == 0
            )

            total_detections = torch.sum(
                preds == 1
            )

            false_positive_rates_by_threshold.append(false_positives/total_negatives)
            precision_rates_by_threshold.append(true_positives/total_detections)
        
        for idx, fpr in enumerate(false_positive_rates_by_threshold):
            if fpr < self.false_positive_rate:
                return precision_rates_by_threshold[idx]


ALL_METRICS = {
    'precision': Precision, 
    'recall': Recall, 
    'auroc': AUROC,
    'precision_froc': PrecisionFROC,
}

def metric_factory(type, kwargs):  
    return ALL_METRICS[type](**kwargs)

def build_metrics_dict(config, prefix):
    d = {}
    for name, kwargs in config.items(): 
        kwargs = OmegaConf.to_object(kwargs)
        type = kwargs.pop('type')
        metric = metric_factory(type, kwargs)
        d[f'{prefix}/{name}'] = metric
        
    return torch.nn.ModuleDict(d)