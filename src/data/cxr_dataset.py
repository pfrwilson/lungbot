import einops
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import os
import numpy as np
from skimage.color import rgb2hsv, gray2rgb

SEEDS = {
    'test': 1, 
    'val': 2
}

class CXRDataset(Dataset):
    
    def __init__(
            self, 
            root, 
            split='train', 
            resample_val=False,
            ignore_negatives=True, 
            to_tensor=True
        ):
        
        super().__init__()
        
        self.root = root
        self.split = split
        self.to_tensor = to_tensor
        
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
        return len(self.idx_df)
    
    def __getitem__(self, idx):
        
        try:
            img_name = self.idx_df.loc[idx, 'img_name']
        except KeyError:
            raise IndexError    
            
        item_metadata = self.metadata.loc[self.metadata.img_name == img_name]
        
        pixel_values = self.__read_mdh_to_numpy(
            os.path.join(
                self.root, 
                'images', 
                img_name)
        )
        
        bounding_boxes = []
        for row in item_metadata.iloc:
            d = dict(row)
            d.pop('img_name')
            bounding_boxes.append(d)
        
        max_ = np.max(pixel_values)
        min_ = np.min(pixel_values)
        pixel_values = (pixel_values - min_)/(max_ - min_)
        
        bounding_boxes = pd.DataFrame(bounding_boxes)
        pixel_values = gray2rgb(pixel_values)
        
        if self.to_tensor:
            values = bounding_boxes[['x', 'y', 'width', 'height']].values
            bounding_boxes = torch.tensor(values)
        
            pixel_values = einops.rearrange(
                pixel_values, 'h w c -> c h w', 
                h = 1024, 
                w = 1024, 
                c = 3
            )
            pixel_values = torch.tensor(pixel_values)
        
        return pixel_values, bounding_boxes      
    
    def compute_box_statistics(self):
        if self.split != 'train':
            raise ValueError('Computing statistics on non-training set' 
                             'destroys statistical validity')
        
        all_boxes = []
        for _, bounding_boxes in self:
            for bounding_box in bounding_boxes.iloc:
                all_boxes.append(bounding_box)

        box_table = pd.DataFrame(all_boxes)
        
        box_tables = {
            'left_lung': box_table.loc[box_table['x'] <= 512],
            'right_lung': box_table.loc[box_table['x'] > 512]
        }
        
        statistics = {}
        for name, table in box_tables.items():
            
            statistics[name] = {}
            statistics[name]['mean'] = \
                table[['x', 'y', 'width', 'height']].mean().values
            statistics[name]['cov_matrix'] = \
                table[['x', 'y', 'width', 'height']].cov().values
            
        return statistics    
    
    def get_box_distribution(self):
        """ Returns a function which can be called to 
        sample a bounding box matching the statistics of the
        true dataset bounding boxes"""
        
        statistics = self.compute_box_statistics()
        
        def sample():
            lung = np.random.randint(0, 2)
            if lung == 0:
                return np.random.multivariate_normal(
                mean = statistics['left_lung']['mean'], 
                cov = statistics['left_lung']['cov_matrix']
            )
            else:
                return np.random.multivariate_normal(
                mean = statistics['right_lung']['mean'], 
                cov = statistics['right_lung']['cov_matrix']
            )
            
        return sample
    
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
    
    
    
