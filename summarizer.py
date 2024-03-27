import torch
from torch.utils.data import DataLoader

from utils.datasets import SummarizationDataset
from utils.config import TransformerConfig
from utils.models import Transformer
from utils.summarize import next_word, summarize


cfg = TransformerConfig()
train_dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cpu"

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
model.load_state_dict(torch.load("summarizer.pt"))
model.train(False)
model = model.to(device)

output = torch.tensor(list(map(train_dt.encoder, ["[SOS]"]))).unsqueeze(0)

out = summarize(
    model,
    "Sleek design. Powerful performance. Intuitive interface. Your everyday tasks just got a whole lot easier.",
    output,
    50,
    train_dt.encoder,
    train_dt.decoder,
    device=device,
)

print(out)
