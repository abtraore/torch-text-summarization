class TransformerConfig:
    # Transformer
    n_blocks = 2
    n_heads = 2
    d_model = 128  # Embedding size
    fully_connected_dim = 128
    max_position_encoding = 256
    max_seq_length_input = 150
    max_seq_length_target = 50
    dropout_rate = 0.1

    # Training
    epochs = 30
    batch_size = 64
    weights_path = "summarizer.pt"
