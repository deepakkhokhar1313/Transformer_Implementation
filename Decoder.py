import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoder import EncoderBlock
from MultiHead_Attention import MultiHeadAttention
from Positional_Embedding import PositionalEmbedding
from Word_Embedding import WordEmbedding


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_fact= 4,n_head= 8):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim,n_head=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)

        self.encoder_block = EncoderBlock(embed_dim,expansion_fact,n_head)

    def forward(self, key, query, value, mask):

        attention = self.attention(query, query,query,mask)
        query = self.dropout(self.norm(attention + query))

        out = self.encoder_block(key, query, value)

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2,
                 expansion_fact=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = WordEmbedding(target_vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            DecoderBlock(embed_dim, expansion_fact, n_heads) for _ in range(num_layers)
        )

        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, encoder_out, mask):
        word_emd_out = self.word_embedding(x)
        positional_emd_out = self.positional_embedding(word_emd_out)

        decoder_out = self.dropout(positional_emd_out)

        for layer in self.layers:
            decoder_out = layer(encoder_out, decoder_out, encoder_out, mask)

        out = F.softmax(self.fc_out(decoder_out))
        
        return out