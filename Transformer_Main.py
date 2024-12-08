import torch
import torch.nn as nn

from Decoder import TransformerDecoder
from Encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_len, 
                 num_layers = 2, expansion_fact=4, n_head = 8):
        super(Transformer, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(seq_len,src_vocab_size,
                                          embed_dim, num_layers, expansion_fact,
                                          n_head)
        self.decoder = TransformerDecoder(target_vocab_size,embed_dim, seq_len,
                                          num_layers, expansion_fact,n_head)
        
    def make_target_mask(self, target_sequence):
        batch_size, target_seq_len = target_sequence.shape

        target_mask = torch.tril(torch.ones((target_seq_len,target_seq_len))).expand(
            batch_size,1,target_seq_len,target_seq_len
        )
        return target_mask
    
    def forward(self, src, target):
        target_mask = self.make_target_mask(target)
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(target,encoder_out, target_mask)

        return decoder_out
    
    # For Inferencing purpose
    def decode(self, src, target):
        target_mask = self.make_target_mask(target)
        encoder_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0],src.shape[1]

        out = target

        for i in range(seq_len):
            out = self.decoder(out, encoder_out,target_mask)
            out = out[:,-1,:]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)
        
        return out_labels