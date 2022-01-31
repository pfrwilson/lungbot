import torch.nn as nn
import einops

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        
        self.l1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, x):
        
        b, n, d = x.shape
        
        x = self.l1(x)
        
        # fold num_rois dimension into batch dimension for batch norm
        x = einops.rearrange(
            x, 
            'b n d -> (b n) d',
            b=b, n=n, d=self.hidden_size
        )
        
        x = self.bn1(x)
        x = self.relu(x)
        
        # unfold
        x = einops.rearrange(
            x, 
            '(b n) d -> b n d', 
            b=b, n=n, d=self.hidden_size
        )
        
        x = self.l2(x)
        
        return x