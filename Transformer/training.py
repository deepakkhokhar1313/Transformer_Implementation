import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import Transformer
from config import Config as config
from utils import log_info, log_error
import time
from nltk.translate.bleu_score import corpus_bleu
import numpy as np


class Trainer:
    def __init__(self, train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, device, tgt_vocab):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            config.model_dim,
            config.num_heads,
            config.ff_dim,
            config.dropout,
            config.num_encoder_layers,
            config.num_decoder_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.tgt_vocab = tgt_vocab  # Store the target vocabulary for BLEU calculation

    def _calculate_accuracy(self, output, target):
      """Calculates accuracy for a batch ignoring padding tokens."""
      predicted_classes = torch.argmax(output, dim=-1)
      mask = target != 0 # mask out padding tokens
      correct_predictions = (predicted_classes == target) * mask
      accuracy = correct_predictions.sum().float() / mask.sum().float()
      return accuracy
      

    def _translate_batch(self, output):
        """Translates the predicted output to text using the vocabulary."""
        predicted_tokens = torch.argmax(output, dim=-1)
        predicted_sentences = []
        for tokens in predicted_tokens:
            sentence = [self.tgt_vocab.itos[token.item()] for token in tokens if token.item() != 0] # Remove padding tokens
            predicted_sentences.append(sentence)
        return predicted_sentences


    def train(self):
        log_info("Starting training")
        start_time = time.time()

        for epoch in range(config.epochs):
            self.model.train()  # set to train mode
            epoch_loss = 0
            epoch_accuracy = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch"):
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]

                self.optimizer.zero_grad()
                output = self.model(src_batch, tgt_input)
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm) # Gradient clipping
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Calculate and accumulate training accuracy for batch
                accuracy = self._calculate_accuracy(output, tgt_output)
                epoch_accuracy += accuracy.item()
                

            avg_train_loss = epoch_loss / len(self.train_dataloader)
            avg_train_accuracy = epoch_accuracy / len(self.train_dataloader)
            val_loss, val_accuracy, bleu_score = self.evaluate()
            log_info(
                f'Epoch: {epoch+1}, '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Train Accuracy: {avg_train_accuracy:.4f}, '
                f'Validation Loss: {val_loss:.4f}, '
                f'Validation Accuracy: {val_accuracy:.4f}, '
                f'BLEU Score: {bleu_score:.4f}'
            )

        end_time = time.time()
        training_time = end_time - start_time
        log_info(f"Training complete in {training_time/60:.2f} minutes")


    def evaluate(self):
        """Evaluates the model on the validation set"""
        log_info("Evaluating model")
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        all_predicted_sentences = []
        all_target_sentences = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                output = self.model(src_batch, tgt_input)
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # Calculate and accumulate validation accuracy
                accuracy = self._calculate_accuracy(output, tgt_output)
                total_accuracy += accuracy.item()
                
                # store translations for BLEU score
                predicted_sentences = self._translate_batch(output) # translated sentences for bleu score calculation
                
                target_sentences = []
                for tokens in tgt_output: # convert the targets to text
                  sentence = [self.tgt_vocab.itos[token.item()] for token in tokens if token.item() != 0]
                  target_sentences.append(sentence)
                
                all_predicted_sentences.extend(predicted_sentences)
                all_target_sentences.extend(target_sentences)

        avg_val_loss = total_loss / len(self.val_dataloader)
        avg_val_accuracy = total_accuracy / len(self.val_dataloader)
        bleu_score = corpus_bleu([[ref] for ref in all_target_sentences], all_predicted_sentences)
        return avg_val_loss, avg_val_accuracy, bleu_score