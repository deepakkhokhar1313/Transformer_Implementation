# inference.py
import torch
import torch.nn.functional as F
from config import Config as config
from utils import log_info, log_error

class Translator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def translate(self, text, max_length = config.max_seq_length):
        """Translates the given text using a greedy approach."""
        log_info(f"Translating: {text}")
        try:
          self.model.eval() # set to evaluation mode
          with torch.no_grad():
            src_ids = self.tokenizer.encode(text, out_type = int)
            src_ids = self._pad_or_truncate(src_ids).unsqueeze(0).to(self.device) # add batch dimension (1, seq_length)
            enc_out = self.model.encoder(src_ids, self.model.make_src_mask(src_ids)) # get encoder output

            tgt_ids = [self.tokenizer.bos_id()]
            for _ in range(max_length):
              tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(self.device)
              tgt_mask = self.model.make_tgt_mask(tgt_tensor)
              output = self.model.decoder(tgt_tensor, enc_out, self.model.make_src_mask(src_ids), tgt_mask)
              output = F.softmax(output[:, -1, :], dim = -1)
              next_token = torch.argmax(output, dim=-1).item()
              if next_token == self.tokenizer.eos_id():
                break
              tgt_ids.append(next_token)

            translated_text = self.tokenizer.decode(tgt_ids)
            log_info(f"Translation: {translated_text}")
            return translated_text
        except Exception as e:
          log_error(f"Error during translation {e}")
          raise

    def _pad_or_truncate(self, token_ids):
      """Pads or truncates the sequence to max_seq_length"""
      token_ids = token_ids[: config.max_seq_length - 2]
      token_ids = [self.tokenizer.bos_id()] + token_ids + [self.tokenizer.eos_id()]
      padding_length = config.max_seq_length - len(token_ids)
      token_ids.extend([self.tokenizer.pad_id()] * padding_length)
      return torch.tensor(token_ids)