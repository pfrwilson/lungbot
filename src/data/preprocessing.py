
# TODO 

def preprocessor_factory(config):
    if config.equalize_hist:
        return torchvision.transforms.functional.equalize
