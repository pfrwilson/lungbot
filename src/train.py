from re import T
from omegaconf import DictConfig
from pytorch_lightning import Trainer


from .data import CXRDataModule
import torch

from .data import CXRDataModule
from .models import RPNSystem
from .models.metric import DetectionMetric

def train(config: DictConfig): 
    
    metric = DetectionMetric(
        **config.metrics
    )
    
    rpn = RPNSystem(metric, **config.model.rpn)
    
    datamodule = CXRDataModule(
        **config.data
    )
    
    trainer = Trainer(
        **config.trainer
    )
    
    trainer.fit(rpn, datamodule)
    
    
    