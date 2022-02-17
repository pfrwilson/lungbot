
from src.data.preprocessing import preprocessor_factory
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from einops import repeat

from .cxr_dataset import CXRDataset

class CXRDataModule(LightningDataModule):
    
    def __init__(self, root: str, batch_size: int, preprocessing_equalize_hist=False, preprocessing = None):
        
        self.root = root
        self.batch_size = batch_size
        self.collate_fn = lambda items: items   # return a list of pairs pixel_values, true_boxes
        self.preprocessing = preprocessor_factory(preprocessing_equalize_hist)
        self.transform = Compose([
            self.preprocessing, 
            Resize(1024), 
            ToTensor(), 
            Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]), 
            Lambda(lambda pixel_values : repeat(pixel_values, 'c h w -> 1 c h w'))
        ])
        
    def setup(self, stage: Optional[str] = None) -> None:
        
        self.train_ds = CXRDataset(self.root, split='train',
                                   transform=self.transform)
        self.val_ds = CXRDataset(self.root, split = 'val',
                                 transform=self.transform)
        self.test_ds = CXRDataset(self.root, split='test', 
                                  transform=self.transform)

    def train_dataloader(self):
        #print(self.transform)
        return DataLoader(
            self.train_ds, 
            batch_size = self.batch_size, 
            collate_fn = self.collate_fn
        )      
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size = self.batch_size, 
            collate_fn = self.collate_fn
        )    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size = self.batch_size, 
            collate_fn = self.collate_fn
        )    
