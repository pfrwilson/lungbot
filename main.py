import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    
    from src.train import train
    from src import utils

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.get("disable_warnings"):
        import warnings
        print("Disabling warnings.")
        warnings.filterwarnings('ignore')
    
    return train(config)


if __name__ == '__main__':
    main()
    
    