import torchvision

# TODO 

def preprocessor_factory(config):
    if config.get('equalize_hist'):
        return torchvision.transforms.functional.equalize
