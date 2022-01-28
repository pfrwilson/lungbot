
import torch
from .densenet import DenseNet121

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14

def build_chexnet():
            
    chexnet = DenseNet121(N_CLASSES)

    ckpt = torch.load(CKPT_PATH)
    chexnet.load_state_dict(ckpt['state_dict'])

    return chexnet

chexnet = build_chexnet()


