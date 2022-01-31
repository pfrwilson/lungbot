from ast import Index
import torch
from torch.utils.data import Dataset
import numpy as np
import skimage
from skimage.exposure import rescale_intensity

class CXRDataset(Dataset):
    
    def __len__(self): 
        return 1024
    
    def __getitem__(self, idx):
        
        if idx not in range(len(self)):
            raise IndexError
    
        background = np.abs(np.random.randn(1024, 1024, 3))
        img = background
        
        num_boxes = np.random.randint(1, 4)
        
        bounding_boxes = []
        for i in range(num_boxes):
            x = np.random.randint(0, 1023)
            y = np.random.randint(0, 1023)
            w = np.random.randint(5, 200)
            h = np.random.randint(5, 200)
            
            bounding_boxes.append({
                'height': h, 
                'width': w, 
                'x': x, 
                'y': y
            })
            
            img[x:x+w, y:y+h] = 1
            
        return img, bounding_boxes
    
    
    
