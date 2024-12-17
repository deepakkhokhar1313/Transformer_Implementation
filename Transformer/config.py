# config.py
class Config:
    # Model parameters
    model_dim = 256 # Dimension of the model's internal representations.
    num_heads = 8  # Number of attention heads in the multi-head attention mechanism.
    ff_dim = 2048   # Dimension of the feedforward network in the transformer.
    dropout = 0.1  # Dropout rate for regularization.
    num_encoder_layers = 4 # Number of encoder layers in the encoder stack.
    num_decoder_layers = 4 # Number of decoder layers in the decoder stack.

    # Training parameters
    batch_size = 16  # Number of samples in a batch during training.
    learning_rate = 0.0001  # Learning rate for the optimizer.
    epochs = 5    # Number of training epochs.
    clip_grad_norm = 1.0  # Gradient clipping to avoid exploding gradients
    beam_size = 4 # Beam size for beam search during inference

    # Data parameters
    max_seq_length = 128 # Maximum length of sequence after padding.
    vocab_size = 10000 # Vocab size for tokenization.

    # Tokenizer parameters
    pad_token = "[PAD]"
    unk_token = "[UNK]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    # Logging parameters
    log_level = "INFO" # Logging level
    log_file = "training.log" # Log file name

# config = Config()