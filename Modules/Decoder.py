import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import PoissonNLLLoss

from .MultiHeadAttention import MultiHeadAttention
from .Block import Block

class Decoder(nn.Module):
    """
    add another attention between encoder's out and decoder
    add one more normalize layer
    """
    def __init__(self, d_model:int, q:int, v:int, h:int, dropout:float = 0.3) -> None:
        super().__init__()

        self._selfAttention = MultiHeadAttention(d_model, q, v, h)
        self._encoderDecoderAttention = MultiHeadAttention(d_model, q, v, h)
        self._feedforward = Block(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, memory:torch.Tensor) -> torch.Tensor:

        out = self._selfAttention(query=x, key=x, value=x, mask="subsequent")
        out = self._dropout(out)
        out = self._layerNorm1(out + x)

        out1 = self._encoderDecoderAttention(query=x, key=x, value=memory)
        out1 = self._dropout(out1)
        out1 = self._layerNorm2(out1 + out)

        out2 = self._feedforward(out1)
        out2 = self._dropout(out2)
        out2 = self._layerNorm3(out2 + out1)

        return out2
    