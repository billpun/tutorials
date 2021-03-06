import torch
import torch.nn as nn 
from src.embeddings import Embedder, PositionalEncoder
from src.layers import Norm, EncoderLayer, DecoderLayer
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, vocab_size, K, N, H, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, K)
        self.pe = PositionalEncoder(K, dropout=dropout)
        self.layers = get_clones(EncoderLayer(K, H, dropout), N)
        self.norm = Norm(K)
        
        
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
        
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, K, N, H, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, K)
        self.pe = PositionalEncoder(K, dropout=dropout)
        self.layers = get_clones(DecoderLayer(K, H, dropout), N)
        self.norm = Norm(K)
        
        
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
        

class Transformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, K, N, H, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, K, N, H, dropout)
        self.decoder = Decoder(trg_vocab, K, N, H, dropout)
        self.out = nn.Linear(K, trg_vocab)
        
        
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
        

def get_model(src_vocab, trg_vocab, K, H, N, dropout):
    
    assert K % H == 0
    assert dropout < 1

    model = Transformer(src_vocab, trg_vocab, K, N, H, dropout)
       
    #if opt.load_weights is not None:
    #    print("loading pretrained weights...")
    #    model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    #else:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 

    #if opt.device == 0:
    #    model = model.cuda()
    return model
    
