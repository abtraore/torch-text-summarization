class TransformerConfig:
    n_blocks = 2
    n_heads = 2
    n_units = 128
    fully_connected_dim = 256
    max_position_encoding_input = 256
    max_position_target_input = 256
    dropout_rate = 0.1
    # training
    epochs = 5
    batch_size = 64
