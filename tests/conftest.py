import os

import pytest
from skimage.color import gray2rgb

ROOT = '/home/ayesha/Desktop/node21/cxr_images/proccessed_data'

@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def cxr_dataset(root):
    from src.data.datasets.cxr_dataset import CXRDataset
    return CXRDataset(root, split=None)


@pytest.fixture
def sample_img(cxr_dataset):
    img = cxr_dataset[0][0]
    return img