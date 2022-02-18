import torchvision
from torchvision import transforms as T
from einops import repeat


def preprocessor_factory(config):

    transforms = []
    
    if config.equalize_hist:
        transforms.append(T.functional.equalize)

    transforms.append(T.Resize(config.chexnet_input_resolution))

    transforms.append(
        T.Compose([
            T.ToTensor(), 
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]), 
            T.Lambda(lambda pixel_values : repeat(pixel_values, 'c h w -> 1 c h w'))
        ])
    )
    
    return T.Compose(transforms)
    




