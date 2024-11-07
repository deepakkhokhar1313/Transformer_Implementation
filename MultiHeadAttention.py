import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 512, n_head = 8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.single_head_dim = int(self.embed_dim / self.n_head)

        # Initilzation of query, key and value matrix(Trainable parameters)
        self.query_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias= False)
        self.key_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias= False)
        self.value_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias= False)
        self.out = nn.Linear(self.n_head * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask = None):
        # key or query or value vector = (batch size, sequence length, embedding dimensions)
        batch_size = key.size(0) #let consider 32
        seq_length = key.siz(1)  #let consider 10
        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # so now vector shape = (32*10*512)
        key = key.view(batch_size, seq_length, self.n_head,self.single_head_dim) #(32*10*8*64)
        query = query.view(batch_size, seq_length_query, self.n_head,self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_head,self.single_head_dim)

        k = self.key_mat(key)
        q = self.query_mat(query)
        v = self.value_mat(value)
        
        # transpose of k
        K_trans = k.transpose(-1,-2)

        product = torch.matmul(q, K_trans)
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))
        
        # Scalling by sqrt(dk)
        product = product / math.sqrt(self.single_head_dim)

        # Applying softmax 
        scores = F.softmax(product, -1)

        # dot product with value matix
        scores = torch.matmul(product, v)

        # concatented output
        concat = scores.transpose(1,2).contiguous().view(batch_size,seq_length_query,self.single_head_dim*self.n_head)

        output = self.out(concat)

        return output


