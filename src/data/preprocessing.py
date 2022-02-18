import torchvision
from torch import Tensor
from torch.nn import Conv2d

# TODO 

def preprocessor_factory(equalize_hist, preprocessing_sharpen):
    transforms = []
    if equalize_hist:
        transforms += [torchvision.transforms.functional.equalize]
    if preprocessing_sharpen:
        filter = Tensor([[1, 0, -1], 
                                [2, 0, -2], 
                                [1, 0, -1]])
        transforms += [Conv2d(filter)]
    return transforms
