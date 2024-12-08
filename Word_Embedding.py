import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(WordEmbedding,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim)
    
    def forward(self,x):
        # x = input vector
        out = self.embed(x)
        # out = embedded vector
        return out
