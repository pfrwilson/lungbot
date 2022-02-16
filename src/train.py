from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .data import CXRDataModule
from .models.rpn_module import RPNModule, RPNModuleConfig

import wandb


def train(config: DictConfig): 
    
    wandb.init(
        project='lungbot',
        config=OmegaConf.to_object(config),
    )
    
    rpn = RPNModule(
        config.model.rpn_module_config
    )
    
    datamodule = CXRDataModule(
        **config.data
    )
    
    logger = WandbLogger(
        project='lungbot', 
        log_model='all'
    )
    
    callbacks = [
        EarlyStopping(monitor='val/precision', patience=10, mode='max'),
        ModelCheckpoint(monitor='val/precision', save_top_k=3, mode='max')
    ]
    
    trainer = Trainer(
        **config.trainer,
        logger=logger, 
        callbacks=callbacks
    )
    
    trainer.fit(rpn, datamodule)
    
    
    