# transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config as config
from utils import log_info, log_error

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
      batch_size = query.shape[0]

      Q = self.wq(query) # (batch_size, seq_length, model_dim)
      K = self.wk(key)   # (batch_size, seq_length, model_dim)
      V = self.wv(value) # (batch_size, seq_length, model_dim)

      # Split the model dimension into multiple heads
      Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
      K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
      V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)

      energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (batch_size, num_heads, seq_length, seq_length)

      if mask is not None:
          energy = energy.masked_fill(mask == 0, -1e10)

      attention = F.softmax(energy, dim=-1) # (batch_size, num_heads, seq_length, seq_length)
      x = torch.matmul(attention, V) # (batch_size, num_heads, seq_length, head_dim)

      x = x.transpose(1, 2).reshape(batch_size, -1, self.model_dim) # (batch_size, seq_length, model_dim)
      x = self.fc_out(x)
      x = self.dropout(x)
      return x

class PositionWiseFeedForward(nn.Module):
  def __init__(self, model_dim, ff_dim, dropout):
    super().__init__()
    self.fc1 = nn.Linear(model_dim, ff_dim)
    self.fc2 = nn.Linear(ff_dim, model_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout):
      super().__init__()
      self.masked_attention = MultiHeadAttention(model_dim, num_heads, dropout)
      self.norm1 = nn.LayerNorm(model_dim)
      self.encoder_attention = MultiHeadAttention(model_dim, num_heads, dropout)
      self.norm2 = nn.LayerNorm(model_dim)
      self.ff = PositionWiseFeedForward(model_dim, ff_dim, dropout)
      self.norm3 = nn.LayerNorm(model_dim)
      self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_out, src_mask, tgt_mask):
      masked_attention_output = self.masked_attention(x, x, x, tgt_mask)
      x = self.norm1(x + self.dropout(masked_attention_output))
      enc_attention_output = self.encoder_attention(x, enc_out, enc_out, src_mask)
      x = self.norm2(x + self.dropout(enc_attention_output))
      ff_output = self.ff(x)
      x = self.norm3(x + self.dropout(ff_output))
      return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, ff_dim, dropout, num_encoder_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_encoder_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
      x = self.embedding(x)
      x = self.pos_encoding(x)
      x = self.dropout(x)
      for layer in self.layers:
        x = layer(x, mask)
      return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, ff_dim, dropout, num_decoder_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_decoder_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(model_dim, vocab_size)


    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        x = self.fc_out(x)
        return x

class PositionalEncoding(nn.Module):
  def __init__(self, model_dim, max_seq_length=config.max_seq_length):
    super().__init__()
    self.model_dim = model_dim
    pe = torch.zeros(max_seq_length, model_dim) # (max_seq_length, model_dim)
    for pos in range(max_seq_length):
      for i in range(0, model_dim, 2):
        pe[pos, i] = math.sin(pos / (10000 ** (i / model_dim)))
        pe[pos, i+1] = math.cos(pos / (10000 ** ((i+1)/ model_dim)))
    self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_seq_length, model_dim)

  def forward(self, x):
      x = x + self.pe[:, : x.size(1), :]
      return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, ff_dim, dropout, num_encoder_layers, num_decoder_layers):
      super().__init__()
      self.encoder = Encoder(src_vocab_size, model_dim, num_heads, ff_dim, dropout, num_encoder_layers)
      self.decoder = Decoder(tgt_vocab_size, model_dim, num_heads, ff_dim, dropout, num_decoder_layers)

    def make_src_mask(self, src):
      src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)
      return src_mask

    def make_tgt_mask(self, tgt):
      tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)
      seq_length = tgt.shape[1]
      nopeak_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool)).to(tgt.device)
      tgt_mask = tgt_mask & nopeak_mask
      return tgt_mask

    def forward(self, src, tgt):
      src_mask = self.make_src_mask(src)
      tgt_mask = self.make_tgt_mask(tgt)
      enc_out = self.encoder(src, src_mask)
      output = self.decoder(tgt, enc_out, src_mask, tgt_mask)
      return output