
from dataclasses import replace
from matplotlib import transforms
import torch
from torch.utils.data import Dataset
import pandas as pd
import einops

from ..processing.roi_sampling import ROISamplerBase
from ..processing.iou_filtering import compute_training_examples


class ROISamplerDatasetForEval(Dataset):
    
    def __init__(self, dataset:Dataset, sampler:ROISamplerBase,
                 num_samples, to_tensor=True):
        
        self.dataset = dataset
        self.sampler = sampler
        self.num_samples = num_samples
        self.to_tensor = to_tensor
    
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        img, true_boxes = self.dataset[idx]
        
        proposed_boxes = self.sampler.sample(img)
        
        proposed_boxes = proposed_boxes.sample(self.num_samples, axis='rows')
        
        if self.to_tensor: 
            img = einops.rearrange(img, 'h w c -> c h w')
            img = torch.tensor(img)
            
            true_boxes = torch.tensor(
                true_boxes[['x', 'y', 'wigth', 'height']].to_numpy()
            )

            proposed_boxes = torch.tensor(
                proposed_boxes[['x', 'y', 'width', 'height']].to_numpy()
            )
            
        return img, true_boxes, proposed_boxes
        

class ROISamplerDatasetForTraining(Dataset):
    
    def __init__(self, dataset: Dataset, sampler: ROISamplerBase, 
                 num_samples, to_tensor=True):
        
        self.dataset = dataset
        self.sampler = sampler
        self.num_samples = num_samples
        self.to_tensor = to_tensor
        
    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, idx):
               
        img, true_boxes = self.dataset[idx]
        
        proposed_boxes = self.sampler.sample(img)
        
        true_boxes, proposed_boxes = compute_training_examples(
            true_boxes, proposed_boxes
        )
        
        # randomly choose 25% positive and 75% negative training examples
        # as described in https://arxiv.org/pdf/1504.08083.pdf 
        
        num_positive = int( 0.25 * self.num_samples )
        num_negative = self.num_samples - num_positive
        
    
        proposed_positives = proposed_boxes.loc[proposed_boxes.labels == 1].sample(num_positive, axis='rows')
        true_box_positives = true_boxes.loc[proposed_positives.index, :]
        
        proposed_negatives = proposed_boxes.loc[proposed_boxes.labels == 0].sample(num_negative, axis='rows')
        true_box_negatives = true_boxes.loc[proposed_negatives.index, :]

        true_boxes = pd.concat([true_box_negatives, true_box_positives], axis=0)
        proposed_boxes = pd.concat([proposed_negatives, proposed_positives], axis=0)
        
        # shuffle
        shuffle_idx = true_boxes.sample(frac=1).index

        true_boxes = true_boxes.loc[shuffle_idx].reset_index(drop=True)
        true_boxes.loc[:, 'label'] = true_boxes['label'].fillna(0)
        
        proposed_boxes = proposed_boxes.loc[shuffle_idx].reset_index(drop=True)
        
        if self.to_tensor: 
            img = einops.rearrange(img, 'h w c -> c h w')
            img = torch.tensor(img)
            
            true_boxes = torch.tensor(
                true_boxes[['x', 'y', 'width', 'height', 'label']].to_numpy()
            )

            proposed_boxes = torch.tensor(
                proposed_boxes[['x', 'y', 'width', 'height']].to_numpy()
            )
        
        return img, true_boxes, proposed_boxes
    

