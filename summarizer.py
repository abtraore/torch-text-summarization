import argparse

import torch
from torch.utils.data import DataLoader

from utils.models import Transformer
from utils.summarize import summarize
from utils.config import TransformerConfig
from utils.datasets import SummarizationDataset


parser = argparse.ArgumentParser("summarizer.py")

parser.add_argument("--input", type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    # Create configuration.
    cfg = TransformerConfig()

    # Instantiate train loader.
    train_dt = SummarizationDataset("data/corpus")
    train_loader = DataLoader(dataset=train_dt, batch_size=cfg.batch_size, shuffle=True)

    # Use cpu for inference.
    device = "cpu"

    # Instantiate model.
    model = Transformer(
        embedding_dim=cfg.d_model,
        input_vocab_size=train_dt.vocab_size,
        target_vocab_size=train_dt.vocab_size,
        input_max_length=cfg.max_position_encoding,
        target_max_length=cfg.max_position_encoding,
        n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
        fully_connected_dim=cfg.fully_connected_dim,
        dropout_rate=cfg.dropout_rate,
        device=device,
    )

    # Load save model and turn eval mode.
    model.load_state_dict(torch.load(cfg.weights_path))
    model.train(False)
    model = model.to(device)

    # Initialize the output with a start of sequence token ([SOS]).
    output = torch.tensor(list(map(train_dt.encoder, ["[SOS]"]))).unsqueeze(0)

    input_text = args.input

    # Start the summarization.
    out = summarize(
        model,
        input_text,
        output,
        cfg.max_seq_length_target,
        train_dt.encoder,
        train_dt.decoder,
        device=device,
    )

    # Print summary.
    print(out.replace("[SOS]", "").replace("[UNK]", "").replace("[EOS]", "").strip())
