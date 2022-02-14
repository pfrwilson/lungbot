from re import T
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from .data import CXRDataModule
import torch

from .data import CXRDataModule
from .models import RPNSystem
from .models.metric import DetectionMetric

def train(config: DictConfig): 
    
    metric = DetectionMetric()
    
    rpn = RPNSystem(metric, **config.model.rpn)
    
    datamodule = CXRDataModule(
        **config.data
    )
    
    logger = WandbLogger(project='lungbot')
    
    trainer = Trainer(
        **config.trainer,
        logger=logger
    )
    
    trainer.fit(rpn, datamodule)
    
    
    