import hydra
from omegaconf import DictConfig

@hydra.main(config_path='configs', config_name='config')
def main(config):
    
    from src.train import train
    from src import utils
    
    from typing import List, Sequence

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
        
    return train(config)


if __name__ == '__main__':
    main()
    
    