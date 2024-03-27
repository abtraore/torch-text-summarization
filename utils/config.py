class TransformerConfig:
    n_blocks = 6
    n_heads = 2
    n_units = 512
    fully_connected_dim = 2048
    max_position_encoding_input = 256
    max_position_target_input = 256
    dropout_rate = 0.1
    # training
    epochs = 20
    batch_size = 64
