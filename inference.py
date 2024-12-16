from typing import List
import torch
from Transformer_Main import Transformer  # Import your Transformer model definition
from DataSetHandler import TranslationDataset  # Import your dataset class for vocab handling

def load_vocab(filepath):
    vocab = {}
    with open(filepath, 'r') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    return vocab

def translate_sentence(input_sentence, model, src_vocab, tgt_vocab, max_seq_len, device):
    """Translate a sentence from source to target language using a trained model."""
    # Reverse target vocabulary for decoding
    rev_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}

    # Tokenize and encode the input sentence
    src_indices = encode_sentence(input_sentence, src_vocab)
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)

    # Prepare target input with <sos> token to start
    tgt_indices = [tgt_vocab['<sos>']]
    tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(device)

    # Generate tokens step by step
    for _ in range(max_seq_len):
        tgt_output = model(src_tensor, tgt_tensor)
        next_token = tgt_output[:, -1, :].argmax(dim=-1).item()  # Get the next token
        tgt_indices.append(next_token)

        if next_token == tgt_vocab['<eos>']:  # Stop if <eos> token is generated
            break

        tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(device)

    # Decode target indices into a sentence
    translated_sentence = decode_sentence(tgt_indices, rev_tgt_vocab)
    return translated_sentence

def encode_sentence(sentence: str, vocab: dict) -> List[int]:
    tokens = sentence.split()  # Tokenize the sentence
    # Add <sos> at the beginning and <eos> at the end of the token list
    tokens = [vocab.get('<sos>', 2)] + [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens] + [vocab.get('<eos>', 3)]
    return tokens

def decode_sentence(indices, rev_vocab):
    """Decode indices back to a sentence."""
    return ' '.join(rev_vocab.get(idx, '<unk>') for idx in indices if idx > 0)

def main():
    # Paths to model and vocab files
    model_path = "Saved_files/transformer_epoch_3.pth"  # Update with the path to your trained model
    src_vocab_path = "Saved_files/src_vocab.txt"  # Update with source vocabulary file path
    tgt_vocab_path = "Saved_files/tgt_vocab.txt"  # Update with target vocabulary file path

    # Load vocabularies
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)

    # Define constants (ensure they match training)
    max_seq_len = 64
    embed_dim = 512
    num_layers = 6
    n_head = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the trained model
    model = Transformer(
        embed_dim=embed_dim,
        src_vocab_size=len(src_vocab),
        target_vocab_size=len(tgt_vocab),
        seq_len=max_seq_len,
        num_layers=num_layers,
        expansion_fact=4,
        n_head=n_head,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Read input sentence from user
    input_sentence = input("Enter a sentence in the source language: ")

    # Translate and output the result
    translated_sentence = translate_sentence(input_sentence, model, src_vocab, tgt_vocab, max_seq_len, device)
    print("Translated sentence:", translated_sentence)

if __name__ == "__main__":
    main()
