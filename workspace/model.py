import math
import torch

from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len,
                 attention_type="full", n_layers=4, n_heads=4,
                 dropout=0.25, attention_dropout=0.25,
                 feed_forward_dimensions=1024):

        super(MusicGenerator, self).__init__()

        self.d_model = d_model

        self.pos_embedding = PositionalEncoding(d_model, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model)

        self.transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            query_dimensions=d_model//n_heads,
            value_dimensions=d_model//n_heads,
            dropout=dropout,
            activation='gelu',
            attention_dropout=attention_dropout
        ).get()

        self.predictor = torch.nn.Linear(d_model, n_tokens)

    def forward(self, x, lengths=None):
        length_mask = None
        if lengths is not None:
            length_mask = LengthMask(lengths, max_len=x.shape[1], device=x.device)

        x = self.value_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)

        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)

        y_hat = self.transformer(x, attn_mask=triangular_mask, length_mask=length_mask)
        y_hat = self.predictor(y_hat)

        return y_hat

class MusicEmotionClassifier(MusicGenerator):
    def forward(self, x, lengths=None):
        length_mask = None
        if lengths is not None:
            length_mask = LengthMask(lengths, max_len=x.shape[1], device=x.device)

        x = self.value_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)

        y_hat = self.transformer(x, length_mask=length_mask)
        y_hat = self.predictor(y_hat)

        # Pool logits considering their lengths
        y_hat = y_hat[range(x.shape[0]), lengths - 1]
        return y_hat
