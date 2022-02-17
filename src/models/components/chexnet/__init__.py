
import torch
from .densenet import DenseNet121
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = os.path.join(
    os.path.dirname(__file__), 
    'model.pth.tar'
)
N_CLASSES = 14

def CheXNet():
            
    chexnet = DenseNet121(N_CLASSES)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = ckpt['state_dict']
    
    # this state dict is from an older version of pytorch with 
    # a different naming convention for state dict keys. 
    # manually fix the keys: 
    def fix_key(key):
        key = key.replace('norm.1.', 'norm1.')
        key = key.replace('norm.2.', 'norm2.')
        key = key.replace('conv.1.', 'conv1.')
        key = key.replace('conv.2.', 'conv2.')
        return key
    for key in list(state_dict.keys()):
        state_dict[fix_key(key)] = state_dict.pop(key)
    
    # this state dict was from a model wrapped in DataParellel module. 
    # remove state dict prefixes to match: 
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    
    chexnet.load_state_dict(ckpt['state_dict'])

    return chexnet.densenet121.features

