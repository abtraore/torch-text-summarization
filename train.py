import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.scheduler import CustomSchedule
from utils.datasets import SummarizationDataset
from utils.config import TransformerConfig
from utils.models import Transformer
from utils.trainer import loops

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create configuration.
cfg = TransformerConfig()

# Instantiate train loader.
train_dt = SummarizationDataset(
    "data/corpus",
    max_enc_seq_length=cfg.max_seq_length_input,
    max_dec_seq_length=cfg.max_seq_length_target,
)
train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)


# Instantiate val loader.
val_dt = SummarizationDataset(
    "data/corpus",
    max_enc_seq_length=cfg.max_seq_length_input,
    max_dec_seq_length=cfg.max_seq_length_target,
    train=False,
)
val_loader = DataLoader(dataset=val_dt, batch_size=cfg.batch_size, shuffle=True)


# Instantiate model.
model = Transformer(
    embedding_dim=cfg.d_model,
    input_vocab_size=train_dt.vocab_size,
    target_vocab_size=train_dt.vocab_size,
    max_position_encoding=cfg.max_position_encoding,
    n_heads=cfg.n_heads,
    n_blocks=cfg.n_blocks,
    fully_connected_dim=cfg.fully_connected_dim,
    dropout_rate=cfg.dropout_rate,
    device=device,
)

model = model.to(device)

optimizer = Adam(params=model.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(cfg.d_model)

loops(
    model, cfg.epochs, train_loader, val_loader, optimizer, scheduler, train_dt, device
)

# Save model weights.
model = model.cpu()
torch.save(model.state_dict(), cfg.weights_path)
