

from .datasets import CXRDataset, ROISamplerDatasetForTraining
from .processing.roi_sampling import RandomSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class CXRBBoxDataModule(LightningDataModule):
    
    def __init__(self, root, batch_size=2, num_proposals_for_train=64, 
                 num_proposals_for_eval=1024):
        
        self.root = root
        self.batch_size = batch_size
        self.num_proposals_for_train = num_proposals_for_train
        self.num_proposals_for_eval = num_proposals_for_eval
        
    def setup(self):
        
        self.box_distribution = CXRDataset(self.root).get_box_distribution()
        
        # training dataset 
        ds = CXRDataset(self.root, split='train')
        sampler = RandomSampler(self.box_distribution, num_samples=100000)
        self.train_ds = ROISamplerDatasetForTraining(
            ds, sampler, self.num_proposals_for_train
        )
        
        # val dataset 
        ds = CXRDataset(self.root, split='val')
        sampler = RandomSampler(self.box_distribution, num_samples=self.num_proposals_for_eval)
        self.val_ds = ROISamplerDatasetForTraining(
            ds, sampler, self.num_proposals_for_eval
        )
    
        # train_dataset
        ds = CXRDataset(self.root, split='test')
        sampler = RandomSampler(self.box_distribution, num_samples=self.num_proposals_for_eval)
        self.test_ds = ROISamplerDatasetForTraining(
            ds, sampler, self.num_proposals_for_eval
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
        
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    
    