from typing import Optional
import numpy as np
import torch

def generate_origin_position(length:int, d_model:int) -> torch.Tensor:
    """
    Generate positional encoding as original paper
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))
    
    return PE

def generate_regular_position(length:int, d_model:int, period: Optional[int] = 24) -> torch.Tensor:
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE