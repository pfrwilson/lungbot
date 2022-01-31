
from numpy import dtype
import pytest
from src.models.components.chexnet import build_chexnet
from tests.conftest import sample_img
import torch
from torchvision.transforms import ToTensor
import einops



@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_chexnet_forward(sample_img, dtype):
    """ 
    Test that the model forward pass of the model can be called
    and that with the appropriate adjustments, it works with either
    single or double precision
    """
    img = ToTensor()(sample_img)
    if dtype == 'float':
        img = img.float()
    
    img = einops.repeat(img, 'c h w -> 1 c h w')
     
    chexnet = build_chexnet()
    
    if dtype == 'double':
        chexnet = chexnet.double()
    
    out = chexnet(img)
    
    assert out.shape == (1, 14)
    
    
