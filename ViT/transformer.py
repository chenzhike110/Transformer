import numpy as np
from torch import nn
from torch.nn import functional as F

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    """merge the last n_dims to a dimension"""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Dot Product Attention
    """
    def __init__(self, dim, num_heads, dropout) -> None:
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # visualization
    
    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : B(batch_size) x S(seq_len)
        D(dim) = H(n_heads) * W(width of head)
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(i, (self.n_heads, -1)).transpose(1, 2) for i in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        self.scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        return h