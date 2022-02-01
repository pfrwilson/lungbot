from ast import Index
import torch
from torch.utils.data import Dataset
import numpy as np
import skimage
from skimage.exposure import rescale_intensity

class DummyCXRDataset(Dataset):

    def __init__(self, length, transform, target_transform):
        self.length = length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): 
        return self.length
    
    def __getitem__(self, idx):
        
        if idx not in range(len(self)):
            raise IndexError

        rng = np.random.RandomState(seed=idx)
        
        background = np.abs(rng.random.randn(1024, 1024, 3))
        img = background
        
        num_boxes = rng.random.randint(1, 4)
        
        bounding_boxes = []
        for i in range(num_boxes):
            x = rng.random.randint(0, 1023)
            y = rng.random.randint(0, 1023)
            w = rng.random.randint(5, 200)
            h = rng.random.randint(5, 200)
            
            bounding_boxes.append({
                'height': h, 
                'width': w, 
                'x': x, 
                'y': y
            })
            
            img[x:x+w, y:y+h] = 1
        
        img = self.transform(img) if self.transform else img
        bounding_boxes = self.target_transform(img) if self.target_transform \
            else bounding_boxes
            
        return img, bounding_boxes
    
    
    
class DummyBoxRegressionDataset(Dataset):
    """A dataset that generates extremely easy bounding box regression examples"""
    
    def __init__(self, length, boxes_per_item, img_size, 
                 transform=None):
        self.img_size = img_size
        self.length = length 
        self.transform = transform 
        self.boxes_per_item = boxes_per_item
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx not in range(len(self)):
            raise IndexError

        rng = np.random.RandomState(seed=idx)
        
        examples = []
        for i in range(self.boxes_per_item):
            
            img = np.zeros( (self.img_size, self.img_size) )
            x = rng.randint(0, self.img_size - 1)
            y = rng.randint(0, self.img_size - 1)
            w = rng.randint(0, self.img_size - 1 - x)
            h = rng.randint(0, self.img_size - 1 - y)
            
            img[ x : x + w, y : y + h] = 1
            
            box = {
                'height': h, 
                'width': w,
                'x': x, 
                'y': y,  
            }
            
            proposed_box = {
                'height': self.img_size,
                'width': self.img_size,
                'x': 0, 
                'y': 0
            }
            
            examples.append((img, box, proposed_box))
            
        return self.transform(examples) if self.transform else examples