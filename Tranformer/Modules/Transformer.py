import torch
import torch.nn as nn

from .Encoder import Encoder
from .Decoder import Decoder
from .PositionEmbedding import generate_origin_position, generate_regular_position

class Transformer(nn.Module):
    def __init__(self, d_input:int, d_model:int, d_output:int, q:int, v:int, h:int, 
                    N:int = 6, dropout:float = 0.3, pe: str = None,pe_period: int = 24) -> None:
        super().__init__()

        self._d_model = d_model
        self.layers_encoding = nn.ModuleList([Encoder(d_model, q, v, h, dropout) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model, q, v, h, dropout) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        position_embedding_function = {
            'origin': generate_origin_position,
            'regular': generate_regular_position,
            None: None
        }
        if pe in position_embedding_function.keys():
            self._generate_position_embedding = position_embedding_function[pe]
            self._pe_period = pe_period

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        K = x.shape[1]
        encoding = self._embedding(x)

        for layer in self.layers_encoding:
            encoding = layer(encoding)
        
        decoding = encoding

        if self._generate_position_embedding is not None:
            position_embedding = self._generate_position_embedding(K, self._d_model)
            position_embedding = position_embedding.to(decoding.device)
            decoding.add_(position_embedding)
        
        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)
        
        output = self._linear(decoding)
        output = self.sigmoid(output)
        return output
