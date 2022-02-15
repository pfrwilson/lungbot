from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from .data import CXRDataModule
from .models.rpn_module import RPNModule, RPNModuleConfig

import wandb


def train(config: DictConfig): 
     
    rpn = RPNModule(
        RPNModuleConfig(**config.model.rpn)
    )
    
    datamodule = CXRDataModule(
        **config.data
    )
    
    logger = WandbLogger(
        **config.logger
    )
    
    callbacks = [
        EarlyStopping(monitor='val/precision', patience=10)
    ]
    
    trainer = Trainer(
        **config.trainer,
        logger=logger, 
        callbacks=callbacks
    )
    
    trainer.fit(rpn, datamodule)
    
    
    