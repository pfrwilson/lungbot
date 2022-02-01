
from gc import callbacks
import einops
from pytorch_lightning import LightningModule, Trainer, Callback
from sklearn.model_selection import PredefinedSplit
from torchvision import transforms as T
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch

def test_bbox_regressor_training():
    """
    Trains the bbox regressor module with single class box regression
    """
    
    from ..stubs.dataset_stubs import DummyBoxRegressionDataset

    def process_input_to_tensors(examples):
        imgs_stack = []
        box_stack = []
        proposed_box_stack = []
        
        for img, box, proposed_box in examples:
            
            box_stack.append([
                box['x'],
                box['y'],
                box['width'], 
                box['height']
            ])
            
            proposed_box_stack.append([
                proposed_box['x'],
                proposed_box['y'],
                proposed_box['width'], 
                proposed_box['height']
            ])
            
            proposed_box_stack.append
            
            imgs_stack.append(img)

        imgs_array = np.array(imgs_stack)
        box_array = np.array(box_stack)
        proposed_box_array = np.array(proposed_box_stack)
        
        return (torch.tensor(imgs_array).float(), 
               torch.tensor(box_array).float(), 
               torch.tensor(proposed_box_array).float())

    ds = DummyBoxRegressionDataset(length=4, boxes_per_item=2, img_size=64,
                                   transform = T.Lambda(process_input_to_tensors))
    
    imgs, boxes, proposed_boxes = ds[0]
    
    assert type(imgs) == torch.Tensor
    assert imgs.shape == (2, 64, 64)
    assert type(boxes) == torch.Tensor
    assert boxes.shape == (2, 4)
    assert type(proposed_boxes) == torch.Tensor
    assert proposed_boxes.shape == (2, 4)
    
    from src.models.components.bbox_regressor import BBoxRegressor
    
    class BBoxRegressionModule(LightningModule):
        
        def __init__(self):
            super().__init__()
            
            self.batch_size = 1
            self.flatten = torch.nn.Flatten(start_dim=2)
            self.bbox_regressor = BBoxRegressor(64*64, 64, 1).float()
            self.loss_fn = torch.nn.MSELoss()
            
        def training_step(self, batch):
            imgs, boxes, proposed_boxes = batch
            
            features = self.flatten(imgs)
            assert features.shape == (1, 2, 64 * 64)
     
            out_dict = self.bbox_regressor(features, proposed_boxes=proposed_boxes)
            
            proposed_boxes = out_dict['proposed_boxes']
            predicted_boxes = out_dict['predicted_boxes']
            
            predicted_boxes = einops.rearrange(predicted_boxes, 'b n c l -> (b n c) l')
            boxes = einops.rearrange(boxes, 'b n l -> (b n) l')

            loss = self.loss_fn(boxes, predicted_boxes)
            
            self.log('train_loss', loss, prog_bar=False, logger=True)
            
            return loss
            
        def train_dataloader(self):
            return DataLoader(
                ds, batch_size = self.batch_size
            )
            
        def configure_optimizers(self):
            return torch.optim.Adam(self.bbox_regressor.parameters())

    losses = []
    class LossCaptureCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
            losses.append(outputs['loss'])
    
    Trainer(max_epochs=100,
            log_every_n_steps=1, callbacks=[LossCaptureCallback()]).fit(BBoxRegressionModule())

    
    # Make sure the loss decreses by a factor of at least 100 throughout training
    assert losses[-1].item() <= losses[0].item() / 100