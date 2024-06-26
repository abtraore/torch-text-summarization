import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_position_encoding: int = 5000):
        super().__init__()

        position = torch.arange(max_position_encoding).unsqueeze(1)

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
        # Adding positional encoding to embedding.
        x += self.pe[:, : x.size(1), :]
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, fully_connected_dim),
            nn.ReLU(),
            nn.Linear(fully_connected_dim, embedding_dim),
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)

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
        embedding_dim,
        vocab_size,
        n_heads=2,
        n_bloks=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        max_position_encoding=150,
        device="cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_heads = n_heads
        self.n_blocks = n_bloks
        self.pos_encoding = PositionalEncoding(
            embedding_dim, max_position_encoding=max_position_encoding
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    fully_connected_dim=fully_connected_dim,
                    dropout_rate=dropout_rate,
                ).to(device)
                for _ in range(self.n_blocks)
            ]
        )

    def forward(self, x, mask=None):

        seq_length = x.shape[1]

        x = self.embedding(x)

        x *= torch.sqrt(torch.tensor(self.embedding_dim))

        x = self.pos_encoding(x)
        x = x[:, :seq_length, :]

        x = self.dropout(x)

        for i in range(self.n_blocks):
            x = self.encoder_layers[i](x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.mha_1 = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.mha_2 = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, fully_connected_dim),
            nn.ReLU(),
            nn.Linear(fully_connected_dim, embedding_dim),
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layer_norm3 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, enc_padding_mask, padding_mask, look_ahead_mask):

        self_mha_out_1, _ = self.mha_1(
            x,
            x,
            x,
            attn_mask=look_ahead_mask,
            key_padding_mask=padding_mask,
            is_causal=True,
        )

        Q1 = self.layer_norm1(x + self_mha_out_1)

        cross_mha_out_2, _ = self.mha_2(
            Q1,
            enc_out,
            enc_out,
            key_padding_mask=enc_padding_mask,
        )

        cross_mha_out_2 = self.layer_norm2(Q1 + cross_mha_out_2)

        ffn_out = self.ffn(cross_mha_out_2)

        ffn_out = self.dropout_ffn(ffn_out)

        decoder_out = self.layer_norm3(cross_mha_out_2 + ffn_out)

        return decoder_out


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocab_size,
        n_heads=2,
        n_bloks=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        max_position_encoding=150,
        device="cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_heads = n_heads
        self.n_blocks = n_bloks
        self.pos_encoding = PositionalEncoding(
            embedding_dim, max_position_encoding=max_position_encoding
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    fully_connected_dim=fully_connected_dim,
                    dropout_rate=dropout_rate,
                ).to(device)
                for _ in range(self.n_blocks)
            ]
        )

    def forward(
        self,
        x,
        enc_out,
        enc_padding_mask=None,
        dec_padding_mask=None,
        look_ahead_mask=None,
    ):

        seq_length = x.shape[1]

        x = self.embedding(x)

        x *= torch.sqrt(torch.tensor(self.embedding_dim))

        x = self.pos_encoding(x)
        x = x[:, :seq_length, :]

        x = self.dropout(x)

        for i in range(self.n_blocks):
            x = self.decoder_layers[i](
                x, enc_out, enc_padding_mask, dec_padding_mask, look_ahead_mask
            )

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_vocab_size,
        target_vocab_size,
        max_position_encoding=150,
        n_heads=2,
        n_blocks=2,
        fully_connected_dim=256,
        dropout_rate=0.1,
        device="cpu",
    ):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate
        self.fully_connected_dim = fully_connected_dim

        self.max_position_encoding = max_position_encoding
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.encoder = Encoder(
            embedding_dim=self.embedding_dim,
            vocab_size=self.input_vocab_size,
            n_heads=n_heads,
            n_bloks=self.n_blocks,
            fully_connected_dim=self.fully_connected_dim,
            max_position_encoding=self.max_position_encoding,
            device=device,
        )

        self.decoder = Decoder(
            embedding_dim=self.embedding_dim,
            vocab_size=self.target_vocab_size,
            n_heads=n_heads,
            n_bloks=self.n_blocks,
            fully_connected_dim=self.fully_connected_dim,
            max_position_encoding=self.max_position_encoding,
            device=device,
        )

        self.classifier = nn.Linear(embedding_dim, target_vocab_size)

    def forward(
        self,
        context,
        target,
        enc_padding_mask=None,
        dec_padding_mask=None,
        dec_look_ahead_mask=None,
    ):

        enc_out = self.encoder(context, enc_padding_mask)
        dec_out = self.decoder(
            target, enc_out, enc_padding_mask, dec_padding_mask, dec_look_ahead_mask
        )

        out = self.classifier(dec_out)

        return out


def create_padding_mask(token_ids):
    mask = token_ids == 0
    return mask


def create_look_ahead_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return mask
