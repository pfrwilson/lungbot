
import pytest
import numpy as np
from src.data.datasets.cxr_dataset import CXRDataset

def test_dataset_instantiation(root):
    
    ds = CXRDataset(root)
    

@pytest.mark.parametrize('expected_keys', 
                         ['height', 'width', 'x', 'y', 'label'])
def test_getitem(root, expected_keys):
    
    ds = CXRDataset(root)
    
    img, boxes = ds[0]
    assert type(img) == np.ndarray    
    assert type(boxes) == list 

    for box_dict in boxes:
        assert type(box_dict) == dict
        assert expected_keys in box_dict.keys()
    
    
    
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
        pass
        

def test_dataset_ignore_positives(root):
    
    ds = CXRDataset(root, ignore_negatives=False)
    
    assert 0 in ds.idx_df['label'].value_counts().index
    
    ds = CXRDataset(root)
    
    assert 0 not in ds.idx_df['label'].value_counts().index
    
    
    