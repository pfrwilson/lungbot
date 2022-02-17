
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig

from typing import Union, Sequence, List

cs = ConfigStore.instance()


@dataclass
class RPNConfig:
    scales: List 
    aspect_ratios: List
    
    nms_threshold: float = 0.7
    
    image_input_size: int = 1024
    feature_map_size: int = 32
    feature_dim: int = 1024
    hidden_dim: int = 256
    


@dataclass
class RPNModuleConfig:
    
    rpn_config: RPNConfig
    
    freeze_chexnet: bool
    lambda_: float
    num_training_examples_per_image: int 
    min_num_positive_examples: int
    positivity_threshold: float 
    negativity_threshold: float 
    lr: float    
    
    metrics_match_threshold: float

cs.store(name='rpn_module_config', node=RPNModuleConfig, group='model')
