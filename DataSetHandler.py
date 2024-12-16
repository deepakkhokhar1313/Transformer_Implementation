from collections import Counter
import os
from typing import List
from torch import tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, max_seq_len: int, vocab_save_path: str = "", is_training_data = True):
        self.max_seq_len = max_seq_len
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

        self.tokenizer = lambda x: x.split()

        # Build vocabularies
        self.src_vocab = self._build_vocab(self.src_sentences)
        self.tgt_vocab = self._build_vocab(self.tgt_sentences)
        
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        # Save vocabularies to files if specified
        if vocab_save_path and is_training_data:
            self._save_vocab(self.src_vocab, os.path.join(vocab_save_path, 'src_vocab.txt'))
            self._save_vocab(self.tgt_vocab, os.path.join(vocab_save_path, 'tgt_vocab.txt'))

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = self.tokenizer(self.src_sentences[idx])
        tgt_tokens = self.tokenizer(self.tgt_sentences[idx])
        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in src_tokens]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in tgt_tokens]
        return tensor(self._pad_tokens(src_indices)), tensor(self._pad_tokens(tgt_indices))

    def _pad_tokens(self, tokens: List[int]) -> List[int]:
        tokens = tokens[:self.max_seq_len]
        return tokens + [0] * (self.max_seq_len - len(tokens))

    def _build_vocab(self, sentences: List[str]) -> dict:
        counter = Counter()
        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            counter.update(tokens)
        
        # Create the vocab, starting from index 2 for actual words
        vocab = {word: idx + 4 for idx, (word, _) in enumerate(counter.most_common())}
        
        # Add special tokens at indices 0, 1, 2, and 3
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        vocab['<sos>'] = 2  # Start of sentence token
        vocab['<eos>'] = 3  # End of sentence token
        
        return vocab
    
    def _save_vocab(self, vocab: dict, file_path: str):
        """Save vocabulary to a text file."""
        with open(file_path, 'w') as f:
            for word, idx in vocab.items():
                f.write(f"{word}\t{idx}\n")