from typing import Optional
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .MultiHeadAttention import MultiHeadAttention
from .Block import Block

class Encoder(nn.Module):
    """
    Encoder from Attention is All You Need

    Parameters:
    d_model:
        Dimension of the input vector
    q:
        Dimension of all query matrix
    v:
        Dimension of all value matrix
    h:
        Number of heads
    """

    def __init__(self, d_model:int, q:int, v:int, h:int, dropout:Optional(float) = 0.3):
        super().__init__()

        self._selfAttention = MultiHeadAttention(d_model, q, v,h)
        self._feedForward = Block(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(p=dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            (batch_size, K, d_model)
        Returns:
            (batch_size, K, d_model)
        """
        out = self._selfAttention(query=x, key=x, value=x)
        out = self._dropout(out)
        x = self._layerNorm1(x + out)

        return x
    
    @property
    def attention_map(self) -> torch.Tensor:
        return self._selfAttention.attention_map
    
    