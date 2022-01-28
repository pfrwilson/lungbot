import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random

SEEDS = {
    'train_test': 1, 
    'train_val': 2
}

class CXRDataset(Dataset):
    
    def __init__(self, root, split='train', resample_train_test=False):
        
        self.root = root
        self.split = split
        self.resample_train_test = resample_train_test
        self.metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)
        self.idx2filepath = self.__get_indexing_splits(split)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        filepath = self.idx2filepath.loc[idx, 'filename']
        item_metadata = self.metadata.loc[self.metadata.img_name == filepath]
        
        pixel_values = self.__read_mdh_to_numpy(
            os.path.join(
                self.root, 
                'images', 
                filepath)
        )
        
        label = item_metadata.loc[item_metadata.index[0], 'label']
        
        bounding_boxes = []
        if label == 1:
            for row in item_metadata.iloc:
                d = dict(row)
                d.pop('label')
                d.pop('img_name')
                bounding_boxes.append(d)
        
        label = {
            'label': label, 
            'bounding_boxes': bounding_boxes
        }
    
        return pixel_values, label
    
    def __get_indexing_splits(self, split):
        
        idx_df = pd.DataFrame(
            os.listdir(os.path.join(self.root, 'images')), 
            columns=['filename']
        ).sort_values(by='filename').reset_index(drop=True)
        
        train_idx, test_idx = train_test_split(idx_df, train_size=0.8, 
                                       random_state=SEEDS['train_test'])
        
        for df in (train_idx, test_idx):
            df.reset_index(drop=True, inplace=True)
        
        train_idx, val_idx = train_test_split(train_idx, train_size=0.9,
                                      random_state=SEEDS['train_val'] 
                                      if self.resample_train_test else random.randint(10, 100))
    
        for df in (train_idx, val_idx):
            df.reset_index(drop=True, inplace=True)
            
        if split == 'train':
            return train_idx
        elif split == 'val':
            return val_idx
        elif split == 'test':
            return test_idx
        else: raise ValueError('split must be one of ["train", "test", "val"].')        
    
    @staticmethod
    def __read_mdh_to_numpy(filename: str):
        
        img = sitk.ReadImage(filename)
        array = sitk.GetArrayFromImage(img)
        
        return array        
        