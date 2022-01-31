
from numpy import dtype
import pytest
from tests.conftest import sample_img
import torch
from torchvision.transforms import ToTensor
import einops


from src.models.components.chexnet import CheXNet
from src.models.components.mlp import MLP


"""
Test that CheXNet instantiates and can be called
"""
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
     
    chexnet = CheXNet()
    
    if dtype == 'double':
        chexnet = chexnet.double()
    
    out = chexnet(img)
    
    assert out.shape == (1, 14)
    
    
def test_mlp():
    
    b = 2     # batch_size
    nroi = 64 # rois per batch
    
    mlp = MLP(
        input_dim=10, 
        hidden_size=10, 
        output_dim=2
    )
    
    sample_input = torch.randn((b, nroi, 10))
    
    out = mlp(sample_input)
    
    assert out.shape == (b, nroi, 2)