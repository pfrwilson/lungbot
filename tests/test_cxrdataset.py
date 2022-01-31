
import pytest
import numpy as np
from src.data.datasets.cxr_dataset import CXRDataset


def test_dataset_instantiation(root):
    ds = CXRDataset(root)
    
    
def test_len(root):
    
    ds = CXRDataset(root)
    assert type(len(ds)) == int
    

@pytest.mark.parametrize('split', ['train', 'test', 'val'])
def test_splits_are_deterministic(root, split):
    
    ds1 = CXRDataset(root, split=split)
    ds2 = CXRDataset(root, split=split)
    
    assert ds1.metadata.equals(ds2.metadata)
    assert ds1.idx_df.equals(ds2.idx_df)
    

@pytest.mark.parametrize('split, expected', 
    [('train', False), ('val', False), ('test', True)]
)
def test_resample_val(root, split, expected):

    ds1 = CXRDataset(root, split=split)
    ds2 = CXRDataset(root, resample_val=True, split=split)

    assert ds1.idx_df.equals(ds2.idx_df) == expected

    
def test_dataset_runthrough(root):
    
    ds = CXRDataset(root)
    for im, labels in ds:
        assert type(im) == np.ndarray
        assert type(labels) == list
        

def test_dataset_with_positives(root):
    
    ds = CXRDataset(root, ignore_negatives=False)
    
    assert 0 in ds.idx_df['label'].value_counts().index