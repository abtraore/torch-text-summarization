import torch
from torch.optim import Adam


from torch.utils.data import DataLoader

from utils.datasets import SummarizationDataset
from utils.config import TransformerConfig
from utils.models import Transformer
from utils.trainer import loops


device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"


cfg = TransformerConfig()
train_dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)

val_dt = SummarizationDataset("data/corpus", train=False)
val_loader = DataLoader(dataset=val_dt, batch_size=cfg.batch_size, shuffle=True)

model = Transformer(
    units=cfg.n_units,
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


class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = torch.tensor(step, dtype=torch.float32)
        d_model = torch.tensor(self.d_model, dtype=torch.float32).float()
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return torch.rsqrt(d_model) * torch.minimum(arg1, arg2)


scheduler = CustomSchedule(cfg.n_units)

loops(model, cfg.epochs, train_loader, val_loader, optimizer, scheduler, device)


model = model.cpu()
torch.save(model.state_dict(), "summarizer.pt")
