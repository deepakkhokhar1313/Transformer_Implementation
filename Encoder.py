import torch
import torch.nn as nn

from MultiHead_Attention import MultiHeadAttention
from Positional_Embedding import PositionalEmbedding
from Word_Embedding import WordEmbedding



class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads = 8):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim,n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=expansion_factor*embed_dim, out_features=embed_dim)
        )

        self.dropout1 =nn.Dropout(0.2)
        self.dropout2 =nn.Dropout(0.2)
    
    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_forward_out = self.feed_forward(norm1_out)
        feed_forward_residual_out = feed_forward_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_forward_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers, expansion_factor=4, n_head=1):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = WordEmbedding(vocab_size,embed_dim)
        self.positional_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([EncoderBlock(embed_dim=embed_dim, 
                                             expansion_factor=expansion_factor,
                                             n_heads=n_head) for i in range(num_layers)])

    def forward(self, x):
        embed_out = self.word_embedding(x)
        out = self.positional_embedding(embed_out)

        for layer in self.layers:
            out = layer(out, out, out)
        
        return out

