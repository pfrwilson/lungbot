from re import T
from omegaconf import DictConfig
from pytorch_lightning import Trainer


from .data import CXRDataModule
import torch

from .data import CXRDataModule
from .models import RPNSystem

def train(config: DictConfig): 
    
    rpn = RPNSystem(**config.rpn)
    
    datamodule = CXRDataModule(**config.data)
    
    trainer = Trainer(**config.trainer)
    
    trainer.fit(rpn, datamodule)
    
    
    