import torch
import torch.nn as nn
from Transformers.FeedForward import FeedForward
from Transformers.MultiHeadAttention import MultiHeadAttention
from Transformers.PositionalEncoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self,d_model:int,h:int,d_ff:int,dropout:float = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x:torch.Tensor, mask:torch.Tensor = None):
        norm_x = self.norm1(x)
        attn_out = self.attention(q=norm_x,k=norm_x,v=norm_x,mask=mask)
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers:int,d_model:int,h:int,d_ff:int,dropout:float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,h,d_ff,dropout) for _ in range (num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self,x:torch.Tensor,mask:torch.Tensor = None):
        for layer in self.layers:
            x = layer(x,mask)
        return self.final_norm(x)