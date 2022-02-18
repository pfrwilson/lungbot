import torchvision

# TODO 

def preprocessor_factory(equalize_hist):
    if equalize_hist:
        return torchvision.transforms.functional.equalize
