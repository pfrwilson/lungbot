
from turtle import width
import numpy as np
import torch
from typing import Optional, Callable
from torch.utils.data import Dataset
import pandas as pd
from selectivesearch import selective_search
from skimage.color import gray2rgb
from src.utils.img_utils import iou
from abc import ABC


class ROISamplerBase(ABC):
    """
    Inherit this class to implement different algorithms for 
    ROI sampling
    """
    
    def __init__(self, base_img):
        self.base_img = base_img
        
    def process(self):
        """
        Return a dataframe with keys ['x', 'y', 'width', 'height']
        specifying the proposed rectangles
        """
        pass
    
    
class SelectiveSearch(ROISamplerBase):
    
    def process(self):
        """Performs a selective search algorithm.

        Args:
            img (np.ndarray): the image to search in numpy rgb format
            scale (int, optional): Scale parameter for the search. Defaults to 1.
            sigma (float, optional): Sigma parameter for the smoothing filter applied 
            before the search. Defaults to 0.8.
            min_size (int, optional): Size below which bounding boxes will not
            be considered. Defaults to 50.

        Returns:
            pd.Dataframe: A table specifying the height, width, x, y, and size of the boxes
        """
        
        img = self.base_img
        
        table = pd.DataFrame(selective_search(img)[1])
        
        def unpack_rectspec(row: pd.Series):
            (x, y, w, h) = row['rect']
            row['x'] = x
            row['y'] = y
            row['width'] = w
            row['height'] = h
        
            return row
        
        table = table.apply(unpack_rectspec, axis='columns').drop(['rect', 'labels'], axis='columns')
        
        return table


class RandomSampler(ROISamplerBase):

    def __init__(
        self, img, size_mean, size_std, num_samples=1000
    ):
        super().__init__(img)
        self.size_mean = size_mean
        self.size_std = size_std
        self.num_samples = num_samples

    def process(self):
        
        img_width, img_height, _ = self.base_img.shape
        
        width = np.random.normal(
            loc=self.size_mean,
            scale=self.size_std,
            size=self.num_samples
        )
        
        height = np.random.normal(
            loc=width, 
            scale=self.size_std, 
            size=self.num_samples
        )
        
        x = np.random.randint(
            low=0, 
            high=img_width, 
            size=self.num_samples
        )
        
        y = np.random.randint(
            low=0, 
            high=img_height, 
            size=self.num_samples
        )

        width = np.where( width <= 0, 1, width)
        height = np.where( height <= 0, 1, height)
        width = np.where( x + width >= img_width, img_width - 1 - x, width,)
        height = np.where( y + height >= img_height, img_height - 1 - y, height,)
        
        table = pd.DataFrame(index=pd.Index(range(self.num_samples), name='box #'))
        table['x'] = x
        table['y'] = y
        table['width'] = width 
        table['height'] = height  
        
        return table     


class ROISamplerDataset(Dataset):
    
    def __init__(
        self, 
        dataset: Dataset, 
        num_samples: int, 
        num_classes: int, 
        roi_sampler: ROISamplerBase,
    ):
        self.dataset = dataset
        self.num_samples = num_samples 
        self.num_classes = num_classes       
        self.roi_sampler = roi_sampler
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self):
        
        img, true_boxes = self.ds[0]
        img = gray2rgb(img)
        
        true_boxes = pd.DataFrame(true_boxes)
        
        proposed_boxes = self.roi_sampler(img)
        
        iou_table = self.compute_iou_table(true_boxes, proposed_boxes)
        
        # =================================================================================================================================================
        # TODO complete the steps of filtering by IoU scores
        # ===================================================================================================================================================

        
    @staticmethod
    def compute_iou_table(true_boxes, proposed_boxes):
        """Returns a table containing ROI values betweeen true boxes (columns)
           and probosed boxes (rows)

        Args:
            true_boxes (pd.DataFrame), proposed_boxes (pd.DataFrame): 
                dataframes containing columns specifying x, y, width, and height of boxes
        """
        
        
        iou_table = pd.DataFrame(index=proposed_boxes.index)
        iou_table.index.name = 'proposed box'
        iou_table.columns.name = 'true box'
        
        for i, true_box_spec in enumerate(true_boxes.iloc):
            
            prop_box_specs = proposed_boxes[['x', 'y', 'width', 'height']].values
            true_box_spec = true_box_spec[['x', 'y', 'width', 'height']].values
            
            iou_ = np.apply_along_axis(
                lambda prop_box_spec : iou(prop_box_spec, true_box_spec),
                axis=1, 
                arr=prop_box_specs,
            )
        
            iou_table[i] = iou_
        
        return iou_table
    


