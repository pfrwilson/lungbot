from omegaconf import DictConfig
from .models.lungbot_module import LungBot
from pytorch_lightning import Trainer
from .data.datamodule import CXRBBoxDataModule
import torch
from torch.utils.data import DataLoader

from .data.datasets import CXRDataset, ROISamplerDatasetForTraining
from .data.processing.roi_sampling import RandomSampler

def train(config: DictConfig): 
    
    model = LungBot(config.model)
    
    datamodule = CXRBBoxDataModule(
        config.data.root
    )
    
    #root = config.data.root
    #ds = CXRDataset(root, split='train')
    #sampler = RandomSampler(ds.get_box_distribution(), num_samples=100000)
    #train_ds = ROISamplerDatasetForTraining(
    #    ds, sampler, 64
    ##)
    ##
    trainer = Trainer(gpus = 1 if torch.cuda.is_available() else 0, 
                      max_epochs = config.training.max_epochs, 
                      )
    
    trainer.fit(model, datamodule)
    
    #return trainer.fit(model, train_dataloaders=DataLoader(dataset=train_ds, batch_size=2)    )
    