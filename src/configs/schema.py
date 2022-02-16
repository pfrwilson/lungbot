
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig

from typing import Union, Sequence

cs = ConfigStore.instance()


@dataclass
class RPNConfig:
    image_input_size: int
    feature_map_size: int 
    feature_dim: int 
    hidden_dim: int 
    scales: Union[list, ListConfig]
    aspect_ratios: Union[list, ListConfig]
    nms_threshold: float


@dataclass
class RPNModuleConfig:
    
    
    scales: list
    aspect_ratios: list
    freeze_chexnet: bool
    lambda_: float
    nms_iou_threshold: float
    num_training_examples_per_image: int 
    min_num_positive_examples: int
    positivity_threshold: float 
    negativity_threshold: float 
    lr: float    

cs.store(name='rpn', node=RPNModuleConfig, group='model')

