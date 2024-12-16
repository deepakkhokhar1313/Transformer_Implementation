import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
import chardet
from torch import tensor

from DataSetHandler import TranslationDataset
from Transformer_Main import Transformer


# 1. Configuration Management
class Config:
    def __init__(self, batch_size, num_epochs, learning_rate, src_vocab_size, tgt_vocab_size, max_seq_length, 
                 embedding_dim, num_heads, num_layers, dropout, data_path):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.data_path = data_path


# 3. Loss Function and Metrics
class LossAndMetrics:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, predictions, targets):
        # Flatten predictions and targets for CrossEntropyLoss
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1)
        return self.criterion(predictions, targets)


# 4. Trainer Class
class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, config: Config):
        self.model = model
        self.config = config

        # DataLoaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.loss_fn = LossAndMetrics()
        self.src_vocab = train_dataset.src_vocab
        self.tgt_vocab = train_dataset.tgt_vocab

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_tokens = 0

        for src, tgt in self.train_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()
            predictions = self.model(src, tgt_input)
            loss = self.loss_fn.compute_loss(predictions, tgt_output)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predicted_tokens = predictions.argmax(dim=-1)
            non_pad_tokens = tgt_output != 0
            correct_predictions += (predicted_tokens == tgt_output).masked_select(non_pad_tokens).sum().item()
            total_tokens += non_pad_tokens.sum().item()

        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        return total_loss / len(self.train_dataloader), accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_tokens = 0
        all_references = []
        all_hypotheses = []

        with torch.no_grad():
            for src, tgt in self.val_dataloader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                predictions = self.model(src, tgt_input)
                loss = self.loss_fn.compute_loss(predictions, tgt_output)
                total_loss += loss.item()

                # Calculate accuracy
                predicted_tokens = predictions.argmax(dim=-1)
                non_pad_tokens = tgt_output != 0
                correct_predictions += (predicted_tokens == tgt_output).masked_select(non_pad_tokens).sum().item()
                total_tokens += non_pad_tokens.sum().item()

                # BLEU Score Calculation
                for i in range(predicted_tokens.size(0)):
                    reference = [self._decode_tokens(tgt_output[i])]
                    hypothesis = self._decode_tokens(predicted_tokens[i])
                    if len(hypothesis) > 0:  # Avoid empty predictions
                        all_references.append(reference)
                        all_hypotheses.append(hypothesis)

        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        bleu_score = self._compute_bleu(all_references, all_hypotheses)
        return total_loss / len(self.val_dataloader), accuracy, bleu_score

    def train(self):
        save_dir = 'Saved_files'
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        for epoch in range(self.config.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy, val_bleu = self.evaluate()

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, BLEU Score: {val_bleu:.4f}")

            # Save model
            torch.save(self.model.state_dict(), f"transformer_epoch_{epoch + 1}.pth")
            # Save model in the specified folder
            model_path = os.path.join(save_dir, f"transformer_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)

    def _decode_tokens(self, indices):
        reverse_vocab = {idx: token for token, idx in self.tgt_vocab.items()}
        return [reverse_vocab.get(idx, '<unk>') for idx in indices.tolist() if idx > 0]

    def _compute_bleu(self, references, hypotheses):
        smooth_fn = SmoothingFunction().method1
        return sum(sentence_bleu(ref, hyp, smoothing_function=smooth_fn) for ref, hyp in zip(references, hypotheses)) / len(references)


# 5. Pipeline Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device =", device)
    data_path = "data/input_data.csv"
    vocab_path = "Saved_files/"

    # Detect file encoding
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        print(f"Detected Encoding: {result['encoding']}")

    # Load data with detected encoding
    data = pd.read_csv(data_path, encoding=result['encoding'], on_bad_lines='skip')

    # Drop rows with missing values
    data = data.dropna(subset=['hindi', 'english'])

    src_sentences = data['hindi'].tolist()
    tgt_sentences = data['english'].tolist()

    # Calculate maximum sequence lengths
    src_max_seq_length = max(len(sentence.split()) for sentence in src_sentences)
    tgt_max_seq_length = max(len(sentence.split()) for sentence in tgt_sentences)

    # Split data into training and validation sets
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src_sentences, tgt_sentences, test_size=0.2, random_state=42
    )

    # Load datasets
    train_dataset = TranslationDataset(src_train, tgt_train, max(src_max_seq_length, tgt_max_seq_length), vocab_path, True)
    val_dataset = TranslationDataset(src_val, tgt_val, max(src_max_seq_length, tgt_max_seq_length), vocab_path, False)

    # Update vocab sizes from dataset
    src_vocab_size = train_dataset.src_vocab_size
    tgt_vocab_size = train_dataset.tgt_vocab_size

    # Initialize configurations
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.001
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    config = Config(
        batch_size, num_epochs, learning_rate, src_vocab_size, tgt_vocab_size,
        max(src_max_seq_length, tgt_max_seq_length), embedding_dim, num_heads, num_layers, dropout, data_path
    )

    # Initialize model
    model = Transformer(embed_dim=config.embedding_dim, src_vocab_size=config.src_vocab_size, target_vocab_size=config.tgt_vocab_size,
                        seq_len=config.max_seq_len, num_layers=config.num_layers, expansion_fact=4, n_head=config.num_heads).to(device)

    # Start training
    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train()
