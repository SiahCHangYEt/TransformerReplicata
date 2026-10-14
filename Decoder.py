import torch
import torch.nn as nn
from Transformers.FeedForward import FeedForward
from Transformers.MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self,d_model:int,d_ff:int,n_head:int,dropout:float = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model,n_head)
        self.cross_attn = MultiHeadAttention(d_model,n_head)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor,enc_output:torch.Tensor,tgt_mask:torch.Tensor = None,memory_mask:torch.Tensor = None):
        norm_x = self.norm_1(x)
        self_attn_out = self.self_attn(norm_x,norm_x,norm_x,mask = tgt_mask)
        x = x + self.dropout(self_attn_out)
        norm_x = self.norm_2(x)
        cross_attn_out = self.cross_attn(q = norm_x, k = enc_output, v = enc_output, mask = memory_mask)
        x = x + self.dropout(cross_attn_out)
        norm_x = self.norm_3(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size:int,d_model:int,d_ff:int,n_head:int,dropout:float = 0.1,num_layers:int=6):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model,d_ff,n_head,dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_projection = nn.Linear(d_model,vocab_size,bias = False)

    def forward(self,x:torch.Tensor,enc_output:torch.Tensor,tgt_mask:torch.Tensor = None,memory_mask:torch.Tensor = None):
        for layer in self.layers:
            x = layer(x,enc_output,tgt_mask,memory_mask)
        x = self.final_norm(x)
        logits = self.out_projection(x)
        return logits

class CasualDecoderLayer(nn.Module):
    def __init__(self,d_model:int,d_ff:int,num_head:int,dropout:float = 0.1):
        super(CasualDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model,num_head)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1: Causal Masked Attention
        norm_x = self.norm_1(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, mask=mask))

        # Sub-layer 2: Feed-Forward
        norm_x = self.norm_2(x)
        x = x + self.dropout(self.ffn(norm_x))
        return x