import torch
from torch.utils.data import DataLoader

from utils.datasets import SummarizationDataset
from utils.config import TransformerConfig
from utils.models import Transformer
from utils.summarize import summarize


# Create configuration.
cfg = TransformerConfig()

# Instantiate train loader.
train_dt = SummarizationDataset("data/corpus")
train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)

# Use cpu for inference.
device = "cpu"


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

# Load save model and turn eval mode.
model.load_state_dict(torch.load("summarizer.pt"))
model.train(False)
model = model.to(device)

# Initialize the out with a start os sequence token ([SOS]).
output = torch.tensor(list(map(train_dt.encoder, ["[SOS]"]))).unsqueeze(0)

input_text = "Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"

out = summarize(
    model,
    input_text,
    output,
    50,
    train_dt.encoder,
    train_dt.decoder,
    device=device,
)

# Print output.
print(out.replace("[SOS]", "").replace("[UNK]", "").replace("[EOS]", "").strip())
