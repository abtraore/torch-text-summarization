import math
import torch

from torch import nn, Tensor

from torch.utils.data import Dataset, DataLoader
from datasets import SummarizationDataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(
        self, units, vocab_size, n_heads=2, fully_connected_dim=256, dropout_rate=0.1
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.mha = nn.MultiheadAttention(embed_dim=units, num_heads=n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(units, fully_connected_dim), nn.Linear(fully_connected_dim, units)
        )

        self.layer_norm1 = nn.LayerNorm(units)
        self.layer_norm2 = nn.LayerNorm(units)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x = self.embedding(x)

        self_mha_out, _ = self.mha(x, x, x)

        skip_connection = self.layer_norm1(x + self_mha_out)

        ffn_out = self.ffn(skip_connection)

        ffn_out = self.dropout_ffn(ffn_out)

        encoder_out = self.layer_norm2(skip_connection + ffn_out)

        return encoder_out


class Encoder(nn.Module):
    def __init__(
        self,
        units,
        vocab_size,
        n_heads=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        max_length=150,
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.n_heads = n_heads
        self.pos_encoding = PositionalEncoding(units, max_len=max_length)

        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = nn.Sequential(
            *[
                EncoderLayer(
                    units=units,
                    vocab_size=vocab_size,
                    n_heads=n_heads,
                    fully_connected_dim=fully_connected_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(3)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.units))
        x = self.pos_encoding(x)
        x = self.dropout(x)

        x = self.encoder_layers(x)

        return x


dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=dt, batch_size=8, shuffle=True)

enc = Encoder(256, dt.vocab_size)

for d, s in train_loader:

    ...
