import math
import torch

from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

class MusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1, bits=32, rounds=4,
                 feed_forward_dimensions=1024,
                 chunk_size=32, masked=True):

        super(MusicGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(d_model//2, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model//2)

        self.transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
            bits=bits,
            rounds=rounds,
            chunk_size=chunk_size,
            masked=masked
        ).get()

        hidden_size = n_heads * d_query
        self.predictor = torch.nn.Linear(hidden_size, n_tokens)

    def forward(self, x, length_mask=None):
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x)
        x = self.pos_embedding(x)

        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        
        if length_mask is not None:
            lengths = (length_mask == 1.).sum(dim=1)
            length_mask = LengthMask(lengths, max_len=x.shape[1], device=x.device)
        
        y_hat = self.transformer(x, attn_mask=triangular_mask, length_mask=length_mask)
        y_hat = self.predictor(y_hat)

        return y_hat
