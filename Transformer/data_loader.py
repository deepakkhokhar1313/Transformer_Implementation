# data_loader.py
import pandas as pd
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import log_info, log_error
import os

class TranslationDataset(Dataset):
    def __init__(self, hindi_texts, english_texts, tokenizer, max_seq_length):
        self.hindi_texts = hindi_texts
        self.english_texts = english_texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.hindi_texts)

    def __getitem__(self, idx):
        hindi_text = self.hindi_texts[idx]
        english_text = self.english_texts[idx]
        hindi_ids = self.tokenizer.encode(hindi_text, out_type=int)
        english_ids = self.tokenizer.encode(english_text, out_type=int)
        hindi_ids = self._pad_or_truncate(hindi_ids)
        english_ids = self._pad_or_truncate(english_ids)

        return torch.tensor(hindi_ids), torch.tensor(english_ids)


    def _pad_or_truncate(self, token_ids):
        """Pads or truncates the sequence to max_seq_length"""
        token_ids = token_ids[: self.max_seq_length - 2]
        token_ids = [self.tokenizer.bos_id()] + token_ids + [self.tokenizer.eos_id()]
        padding_length = self.max_seq_length - len(token_ids)
        token_ids.extend([self.tokenizer.pad_id()] * padding_length)
        return token_ids


class DataLoaderLocal:
    def __init__(self, csv_file, vocab_size, max_seq_length, batch_size):
        self.csv_file = csv_file
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = self._create_tokenizer()
        self.tgt_vocab = self._create_vocab()

    def _create_tokenizer(self):
        """Creates the sentencepiece tokenizer"""
        log_info("Creating tokenizer")
        try:
           if not os.path.exists('tokenizer.model'):
              log_info("Creating tokenizer from scratch")
              df = pd.read_csv(self.csv_file)
              all_texts = df['hindi'].tolist() + df['english'].tolist()
              with open("all_texts.txt", "w", encoding="utf-8") as f:
                for text in all_texts:
                  f.write(str(text) + "\n")

              spm.SentencePieceTrainer.train(
              f'--input=all_texts.txt --model_prefix=tokenizer --vocab_size={self.vocab_size} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_type=bpe'
              )
           sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
           log_info("Tokenizer created successfully")
           return sp
        except Exception as e:
          log_error(f"Error during tokenizer creation {e}")
          raise


    def _create_vocab(self):
        """Creates a vocabulary dictionary from the tokenizer."""
        log_info("Creating vocabulary")
        if not hasattr(self, 'tokenizer'):
          log_error("Tokenizer not initialized")
          raise ValueError("Tokenizer not initialized")
        
        vocab_list = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        itos = {idx: token for idx, token in enumerate(vocab_list)}
        log_info("Vocabulary created")
        class VocabWrapper:
           def __init__(self, itos):
              self.itos = itos
           
        return VocabWrapper(itos=itos)
    
    
    def get_data_loaders(self):
         """Reads csv and returns training and validation dataloaders"""
         log_info("Creating data loaders")
         try:
             df = pd.read_csv(self.csv_file)
             hindi_texts = df['hindi'].tolist()
             english_texts = df['english'].tolist()

             dataset = TranslationDataset(hindi_texts, english_texts, self.tokenizer, self.max_seq_length)

             train_size = int(0.8 * len(dataset))
             val_size = len(dataset) - train_size
             train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

             train_dataloader =  DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
             val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
             log_info("Data loaders created successfully")
             return train_dataloader, val_dataloader
         except Exception as e:
             log_error(f"Error during data loading {e}")
             raise