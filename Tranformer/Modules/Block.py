import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class Block(nn.Module):
    """
    Nerual nework position embedding
    """

    def __init__(self, d_model:int, d_ff:Optional[int] = 2048):
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.relu(self._linear1(x))
        x = self._linear2(x)
        return x