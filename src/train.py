from src.data.preprocessing import preprocessor_factory
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from .data import CXRDataModule
from .models.rpn_module import RPNModule, RPNModuleConfig

import wandb


def train(config: DictConfig): 
    
    wandb.init(
        project='lungbot',
        config=OmegaConf.to_object(config),
        name=config.run_name
    )
    
    rpn = RPNModule(
        config.model.rpn_module_config
    )
    
    datamodule = CXRDataModule(
        **config.data
    )
    
    logger = WandbLogger(
        **config.logger
    )
    
    callbacks = [
        EarlyStopping(**config.callbacks.early_stopping),
        ModelCheckpoint(**config.callbacks.model_checkpoint)
    ]
    
    trainer = Trainer(
        **config.trainer,
        logger=logger, 
        callbacks=callbacks
    )
    
    if config.seed.get("pytorch_lightning_seed"):
        seed_everything(config.seed.pytorch_lightning_seed)
    
    trainer.fit(rpn, datamodule)
    
    
    
