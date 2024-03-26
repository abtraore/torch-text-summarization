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

        k = torch.arange(d_model).unsqueeze(0)

        i = k // 2

        angle_rates = 1 / torch.pow(
            10000, (2 * i) / torch.tensor(d_model, dtype=torch.float)
        )

        pe = position * angle_rates

        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:

        x += self.pe[:, : x.size(1), :]

        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(
        self, units, vocab_size, n_heads=2, fully_connected_dim=256, dropout_rate=0.1
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.mha = nn.MultiheadAttention(
            embed_dim=units, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(units, fully_connected_dim), nn.Linear(fully_connected_dim, units)
        )

        self.layer_norm1 = nn.LayerNorm(units)
        self.layer_norm2 = nn.LayerNorm(units)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, mask):

        self_mha_out, _ = self.mha(x, x, x, key_padding_mask=mask)

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
        n_bloks=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        max_length=150,
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.n_heads = n_heads
        self.n_blocks = n_bloks
        self.pos_encoding = PositionalEncoding(units, max_len=max_length)

        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = [
            EncoderLayer(
                units=units,
                vocab_size=vocab_size,
                n_heads=n_heads,
                fully_connected_dim=fully_connected_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(self.n_blocks)
        ]

    def forward(self, x, mask=None):

        x = self.embedding(x)

        x *= torch.sqrt(torch.tensor(self.units))

        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.n_blocks):
            x = self.encoder_layers[i](x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, units, vocab_size, n_heads=2, fully_connected_dim=256, dropout_rate=0.1
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.mha_1 = nn.MultiheadAttention(
            embed_dim=units, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )

        self.mha_2 = nn.MultiheadAttention(
            embed_dim=units, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(units, fully_connected_dim), nn.Linear(fully_connected_dim, units)
        )

        self.layer_norm1 = nn.LayerNorm(units)
        self.layer_norm2 = nn.LayerNorm(units)
        self.layer_norm2 = nn.LayerNorm(units)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, padding_mask, look_ahead_mask):

        self_mha_out_1, _ = self.mha_1(
            x,
            x,
            x,
            attn_mask=look_ahead_mask,
            is_causal=True,
        )

        Q1 = self.layer_norm1(x + self_mha_out_1)

        self_mha_out_2, _ = self.mha_2(
            Q1,
            enc_out,
            enc_out,
            key_padding_mask=padding_mask,
        )

        self_mha_out_2 = self.layer_norm2(Q1 + self_mha_out_2)

        ffn_out = self.ffn(self_mha_out_2)

        ffn_out = self.dropout_ffn(ffn_out)

        decoder_out = self.layer_norm2(self_mha_out_2 + ffn_out)

        return decoder_out


class Decoder(nn.Module):
    def __init__(
        self,
        units,
        vocab_size,
        n_heads=2,
        n_bloks=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        max_length=150,
    ):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.n_heads = n_heads
        self.n_blocks = n_bloks
        self.pos_encoding = PositionalEncoding(units, max_len=max_length)

        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_layers = [
            DecoderLayer(
                units=units,
                vocab_size=vocab_size,
                n_heads=n_heads,
                fully_connected_dim=fully_connected_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(self.n_blocks)
        ]

    def forward(self, x, enc_out, padding_mask=None, look_ahead_mask=None):

        x = self.embedding(x)

        x *= torch.sqrt(torch.tensor(self.units))

        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.n_blocks):
            x = self.decoder_layers[i](x, enc_out, padding_mask, look_ahead_mask)

        return x


def create_padding_mask(token_ids):
    # All the padding will have 0 as value
    mask = (token_ids == 0).float()
    # Add an extra dimension to allow broadcasting.
    # mask = mask.unsqueeze(1)
    return mask


def create_look_ahead_mask(seq_length):
    mask = torch.tril(torch.ones((seq_length, seq_length)))
    return mask


units = 256

dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=dt, batch_size=8, shuffle=True)

enc = Encoder(units, dt.vocab_size)
dec = Decoder(units, dt.vocab_size, max_length=50)

for d, s in train_loader:

    enc_padding_mask = create_padding_mask(d)

    dec_padding_mask = create_padding_mask(s)
    dec_look_ahead_mask = create_look_ahead_mask(s.shape[1])

    # print(dec_look_ahead_mask.shape)

    out = enc(d, enc_padding_mask)

    out = dec(s, out, None, dec_look_ahead_mask)

    # print(out.shape)

    break
    ...
