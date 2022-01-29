from curses import meta
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np

SEEDS = {
    'test': 1, 
    'val': 2
}


class CXRDataset(Dataset):
    
    def __init__(self, root, split='train', resample_val=False,
                 ignore_negatives=True, transform=None, target_transform=None, 
                 convert_to_float=True):
        
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.convert_to_float = convert_to_float
        
        self.metadata = pd.read_csv(
            os.path.join(root, 'metadata.csv'),
            index_col=0
        )

        idx_df = self.__get_idx_df(root, resample_val)
        
        # add labels
        idx_df = pd.merge(
            idx_df, 
            self.metadata[['img_name', 'label']].drop_duplicates(), 
            on='img_name', 
            how='inner'
        )
        
        # extract split
        if split:
            assert split in ['train', 'val', 'test']
            idx_df = idx_df.loc[idx_df['split'] == split]
        
        if ignore_negatives:
            idx_df = idx_df.loc[idx_df['label'] == 1]
        
        self.idx_df = idx_df.reset_index(drop=True)
        
        
    def __len__(self):
        return len(self.idx2filepath)
    
    def __getitem__(self, idx):
        
        img_name = self.idx_df.loc[idx, 'img_name']
        item_metadata = self.metadata.loc[self.metadata.img_name == img_name]
        
        pixel_values = self.__read_mdh_to_numpy(
            os.path.join(
                self.root, 
                'images', 
                img_name)
        )
        
        label = item_metadata.loc[item_metadata.index[0], 'label']
        
        bounding_boxes = []
        if label == 1:
            for row in item_metadata.iloc:
                d = dict(row)
                d.pop('label')
                d.pop('img_name')
                bounding_boxes.append(d)
        
        if self.convert_to_float:
            max_ = np.max(pixel_values)
            min_ = np.min(pixel_values)
            pixel_values = (pixel_values - min_)/(max_ - min_)
        
        if self.transform:
            pixel_values = self.transform(pixel_values)
        if self.target_transform:
            bounding_boxes = self.target_transform(bounding_boxes)
    
        return pixel_values, bounding_boxes      
    
    @staticmethod
    def __read_mdh_to_numpy(filename: str):
        
        img = sitk.ReadImage(filename)
        array = sitk.GetArrayFromImage(img)
        
        return array        
    
    @staticmethod
    def __get_idx_df(root, resample_val):
        
        idx_df = pd.DataFrame(
                os.listdir(os.path.join(root, 'images')), 
                columns=['img_name']
            ).sort_values(by='img_name').reset_index(drop=True)
        
        idx_df['split'] = pd.Series(dtype='object')

        # generate masks
        
        rng = np.random.RandomState(seed=SEEDS['test'])
        test_mask = rng.rand(len(idx_df)) >= 0.8

        rng = np.random.RandomState(seed=None if resample_val else SEEDS['val'])
        val_mask = np.logical_and(rng.rand(len(idx_df)) >= 0.9, ~test_mask)

        train_mask = np.logical_and(~val_mask, ~test_mask)
        
        idx_df['split'] = idx_df['split'].mask(test_mask, 'test')
        idx_df['split'] = idx_df['split'].mask(val_mask, 'val')
        idx_df['split'] = idx_df['split'].mask(train_mask, 'train')
    
        idx_df.to_csv(
            os.path.join(root, 'split_info.csv'),
            header=True
        )
        
        return idx_df
        
        
