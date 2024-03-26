import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.datasets import SummarizationDataset
from utils.config import TransformerConfig
from utils.models import Transformer
from utils.models import create_look_ahead_mask, create_padding_mask
from utils.loss import mask_loss
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
    input_max_length=train_dt.max_enc_seq_length,
    target_max_length=train_dt.max_dec_seq_length,
    n_heads=cfg.n_heads,
    n_blocks=cfg.n_blocks,
    fully_connected_dim=cfg.fully_connected_dim,
    dropout_rate=cfg.dropout_rate,
    device=device,
)

model = model.to(device)

optimizer = AdamW(params=model.parameters())
scheduler = ReduceLROnPlateau(optimizer, "min", patience=2)

# for context, target in train_loader:

#     context = context.to(device)
#     target = target.to(device)

#     enc_padding_mask = create_padding_mask(context)
#     dec_padding_mask = create_padding_mask(context)
#     dec_look_ahead_mask = create_look_ahead_mask(target.shape[1]).to(device)

#     optimizer.zero_grad()

#     out = model(
#         context, target, enc_padding_mask, dec_padding_mask, dec_look_ahead_mask
#     )

#     loss = mask_loss(out, target)

#     loss.backward()
#     optimizer.step()

#     print(loss.item())

#     # break
#     # ...


loops(model, cfg.epochs, train_loader, val_loader, optimizer, scheduler, device)
