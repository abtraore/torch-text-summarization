class TransformerConfig:
    n_blocks = 5
    n_heads = 2
    d_model = 512  # Embedding size
    fully_connected_dim = 2048
    max_position_encoding = 256
    max_seq_length_input = 150  # TODO: Add to model
    max_seq_length_target = 50  # TODO: Add to model
    dropout_rate = 0.1
    # training
    epochs = 14
    batch_size = 64
    weights_path = "summarizer.pt"
