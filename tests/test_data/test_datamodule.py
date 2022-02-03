

from src.data.datamodule import CXRBBoxDataModule


def test_train_dataloader(root):
    
    datamodule = CXRBBoxDataModule(root)
    
    datamodule.setup()
    
    loader = datamodule.train_dataloader()
    
    batch = next(iter(loader))

    assert batch.shape == None