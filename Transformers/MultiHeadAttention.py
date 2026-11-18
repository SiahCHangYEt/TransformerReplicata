import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):

    def __init__(self,d_model,number_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.number_of_heads = number_of_heads
        self.d_k = d_model // number_of_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        #W_O
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self,q,k,v,mask=None,kv_cache=None,return_attn=False):
        batch_size = q.size(0)
        q_seq_len = q.size(1)
        k_seq_len = k.size(1)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = q.view(batch_size, q_seq_len, self.number_of_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k_seq_len, self.number_of_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, k_seq_len, self.number_of_heads, self.d_k).transpose(1, 2)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat((prev_k, k), dim=2)
            v = torch.cat((prev_v, v), dim=2)
        new_kv_cache = (k, v)
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            # Guarantee mask is boolean and broadcastable:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights,v)
        context = context.transpose(1,2).contiguous().view(batch_size,q_seq_len, self.d_model)
        output = self.out_proj(context)

        if return_attn:
            return output, new_kv_cache, attn_weights
        return output, new_kv_cache

def block_hashes(token_ids, salt = None):
    """Chain-hash a token sequence into per-block keys."""
    BLOCK_SIZE = 16
    hashes, parent = [], hash(salt)

    # Only complete blocks are hashed. A partial tail block is skipped.
    for start in range(0, len(token_ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = tuple(token_ids[start:start + BLOCK_SIZE])
        parent = hash((parent,block))
        hashes.append(parent)

    return hashes




