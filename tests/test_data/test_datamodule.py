

from torch import Tensor
from src.data.datamodule import CXRBBoxDataModule


def test_train_dataloader(root):
    
    datamodule = CXRBBoxDataModule(root)
    
    datamodule.setup()
    
    loader = datamodule.train_dataloader()
    
    batch = next(iter(loader))

    assert type(batch) == list
    assert type(batch[0]) == Tensor
    assert batch[0].shape == (2, 3, 1024, 1024)
    