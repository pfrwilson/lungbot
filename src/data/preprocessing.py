import torchvision
import torch

# TODO 

def preprocessor_factory(equalize_hist, preprocessing_sharpen):
    transforms = []
    if equalize_hist:
        transforms += [torchvision.transforms.functional.equalize]
    if preprocessing_sharpen:
        filter = torch.Tensor([[1, 0, -1], 
                                [2, 0, -2], 
                                [1, 0, -1]])
        conv = torch.nn.Conv2d(1, 1, 3, bias=False)
        with torch.no_grad():
            conv.weight = torch.nn.Parameter(filter)
        transforms += [conv]
    return transforms
