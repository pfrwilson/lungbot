import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='configs', config_name='config')
def main(config):
    
    from src.train import train
    from src import utils
    
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # disable warnings
    if config.get("disable_warnings"):
        import warnings
        print("Disabling warnings.")
        warnings.filterwarnings('ignore')
    
    return train(config)


if __name__ == '__main__':
    main()
    
    