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
train_dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)


# Instantiate val loader.
val_dt = SummarizationDataset("data/corpus", train=False)
val_loader = DataLoader(dataset=val_dt, batch_size=cfg.batch_size, shuffle=True)

# Instantiate model.
model = Transformer(
    units=cfg.d_model,
    input_vocab_size=train_dt.vocab_size,
    target_vocab_size=train_dt.vocab_size,
    input_max_length=cfg.max_position_encoding_input,
    target_max_length=cfg.max_position_encoding_input,
    n_heads=cfg.n_heads,
    n_blocks=cfg.n_blocks,
    fully_connected_dim=cfg.fully_connected_dim,
    dropout_rate=cfg.dropout_rate,
    device=device,
)

model = model.to(device)

optimizer = Adam(params=model.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(cfg.d_model)

loops(model, cfg.epochs, train_loader, val_loader, optimizer, scheduler, device)

# Save model weights.
model = model.cpu()
torch.save(model.state_dict(), "summarizer.pt")
