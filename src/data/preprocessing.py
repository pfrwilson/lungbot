import torchvision
import torch
from einops import repeat

# TODO 

def preprocessor_factory(equalize_hist, preprocessing_sharpen):
    transforms = []
    if equalize_hist:
        transforms += [torchvision.transforms.functional.equalize]
    if preprocessing_sharpen:
        filter = torch.unsqueeze(torch.Tensor([[1, 0, -1], 
                                [2, 0, -2], 
                                [1, 0, -1]]), dim=0)
        conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding='same')
        with torch.no_grad():
            conv.weight = torch.nn.Parameter(filter)
        transforms += [torchvision.transforms.ToTensor(), conv, torchvision.transforms.Lambda(lambda pixel_values : repeat(pixel_values, '1 c h w -> c h w'))]
    return transforms
