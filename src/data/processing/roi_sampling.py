
from turtle import width
from matplotlib.pyplot import axis
import numpy as np
import torch
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
import pandas as pd
from selectivesearch import selective_search
from skimage.color import gray2rgb
from src.utils.img_utils import iou
from abc import ABC, abstractmethod


class ROISamplerBase(ABC):
    """
    Inherit this class to implement different algorithms for 
    ROI sampling
    """
    
    @abstractmethod
    def sample(self, img):
        """
        Return a dataframe with keys ['x', 'y', 'width', 'height']
        specifying the proposed rectangles for the given input image
        """
        pass


class SelectiveSearch(ROISamplerBase):
    
    def sample(self, img):
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
        self, 
        distribution: Callable[..., Tuple[int, int, int, int]], 
        num_samples: int=256, 
    ):
        """Create a random sampler which samples boxes according to a
        specified distribution.

        Args:
            img (np.ndarray): the image from which samples are drawn
            distribution (Callable): a function which returns a box proposal
            [x, y, w, h] from a desired probability distribution.
            num_samples (int, optional): The number of samples to be drawn.
            Defaults to 256.
        """
        self.distribution = distribution
        self.num_samples = num_samples

    def sample(self, img):
        
        samples = [self.distribution() for i in range(self.num_samples)]
        
        samples = pd.DataFrame(samples, columns = ['x', 'y', 'width', 'height'])
        samples.index.name = 'box #'
        
        return samples
    


