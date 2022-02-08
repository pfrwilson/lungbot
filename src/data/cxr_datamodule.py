
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import torch
from .cxr_dataset import CXRDataset


def collate_items(items: List[torch.Tensor]):
    
    pixel_values = [ pixel_values for (pixel_values, _) in items]
    pixel_values = torch.stack(pixel_values, dim=0)
    
    true_boxes = [ true_boxes for (_, true_boxes) in items]

    true_box_indices = []
    for i in range(len(true_boxes)):
        true_box_indices.extend( [i] * len(true_boxes[i]) )
    true_box_indices = torch.tensor( true_box_indices )
    true_boxes = torch.concat(true_boxes, axis=0)
    
    return pixel_values, true_boxes, true_box_indices


class CXRDataModule(LightningDataModule):
    
    def __init__(self, root: str, batch_size: int):
        
        self.root = root
        self.batch_size = batch_size
        self.collate_fn = collate_items
        
    def setup(self, stage: Optional[str] = None) -> None:
        
        self.train_ds = CXRDataset(self.root, split='train')
        self.val_ds = CXRDataset(self.root, split = 'val')
        self.test_ds = CXRDataset(self.root, split='test')

    def train_dataloader(self):
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