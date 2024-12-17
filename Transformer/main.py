# main.py
import torch
import argparse
from data_loader import DataLoader, DataLoaderLocal
from training import Trainer
from inference import Translator
from config import Config as config
from transformer import Transformer
from utils import setup_logger, log_info
import os

def main():
    torch.cuda.empty_cache()

    setup_logger()

    parser = argparse.ArgumentParser(description="Train and translate using a Transformer model.")
    # parser.add_argument("--csv_file", type=str, default="data.csv", help="Path to the CSV file containing the data.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--translate", type=str, help="Text to translate.")
    args = parser.parse_args()

    
    # csv_file = args.csv_file
    csv_file  = os.path.join("data", "input_data.csv")  

    if not os.path.exists(csv_file):
      log_info(f"{csv_file} does not exist.")
      return

    log_info("Starting data loading")
    data_loader = DataLoaderLocal(
        csv_file=csv_file,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
    )
    train_dataloader, val_dataloader = data_loader.get_data_loaders()

    tokenizer = data_loader.tokenizer
    src_vocab_size = tokenizer.get_piece_size()
    tgt_vocab_size = tokenizer.get_piece_size()
    tgt_vocab = data_loader.tgt_vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    if args.train:
        log_info("Starting training process")
        trainer = Trainer(
            train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, device, tgt_vocab
        )
        trainer.train()
        log_info("Training finished")
        torch.save(trainer.model.state_dict(), "trained_model.pth")
        log_info("Trained model saved to trained_model.pth")


    if args.translate:
        log_info("Starting translation")
        model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            config.model_dim,
            config.num_heads,
            config.ff_dim,
            config.dropout,
            config.num_encoder_layers,
            config.num_decoder_layers
        ).to(device)
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        translator = Translator(model, tokenizer, device)
        translated_text = translator.translate(args.translate)
        print(f"Translated Text: {translated_text}")
        log_info("Translation finished")


if __name__ == "__main__":
    main()