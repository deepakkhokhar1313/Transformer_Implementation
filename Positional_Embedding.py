import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            # This loop will show word position in sequene
            for i in range(0,self.embed_dim,2):
                # this loop willupdate values for 
                # each dimension of vector
                pe[pos, i] = math.sin(pos/(10000 **((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos/(10000 **((2 * (i + 1))/self.embed_dim)))
        
        # it add one extra dimension for batch size 
        # at later stage
        pe = pe.unsqueeze(0)
        # This registers the pe tensor as a buffer. 
        # Buffers are tensors that are not considered model parameters 
        # (i.e., they don't get updated by gradient descent), 
        # but they are part of the model and will be moved to the GPU/CPU with the model.
        self.register_buffer('pe',pe)

    def forward(self, x):
        # Scaling the input vector so that they do no have very big valus
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        # ensures that the positional encodings are not involved in backpropagation 
        # (i.e., they are not trainable)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad = False)
        return x