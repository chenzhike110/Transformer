from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention block from Attention is All You Need

    inputs: 
        (batch_size, K, d_model)
    outputs:
        (batch_size, K, d_model)

    Parameters
    ----------
    d_model:
        Dimension of the input vector
    q:
        Dimension of all query matrix
    v:
        Dimension of all value matrix
    h:
        Number of heads
    """

    def __init__(self, d_model:int, q:int, v:int, h:int) -> None:
        super().__init__()

        self._h = h
        # query, keys, value matrices
        self._Wq = nn.Linear(d_model, q*self._h)
        self._Wk = nn.Linear(d_model, q*self._h)
        self._Wv = nn.Linear(d_model, v*self._h)
        
        # output linear function
        self._Wo = nn.Linear(self._h*v, d_model)

        # score placeholder
        self._scores = None

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:Optional[str] = None) -> torch.Tensor:
        K = query.shape[1]

        queries = torch.cat(self._Wq(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._Wk(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._Wv(value).chunk(self._h, dim=-1), dim=0)

        self._scores = torch.bmm(queries, keys.transpose(1,2))/np.sqrt(K)
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))
        self._scores = F.softmax(self._scores)

        attention = torch.bmm(self._scores, values)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self._Wo(attention_heads)

        return self_attention
        
    
    @property
    def attention_map(self) -> torch.Tensor:
        """
        Attention map which indicates the relation between words
        """
        if self._scores is None:
            raise RuntimeError("No Forward before")
        return self._scores