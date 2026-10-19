# In Pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding.Embeddings import TransformerEmbedding
from Transformers.PositionalEncoding import PositionalEncoding
from Decoder import CasualDecoderLayer

class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_head: int = 4, d_ff: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CasualDecoderLayer(d_model, d_ff, n_head, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional weight tying (improves small model convergence):
        self.head.weight = self.emb.lut.weight

    def forward(self, idx: torch.Tensor, mask: torch.Tensor = None, kv_caches: list = None, start_pos: int = 0):
        x = self.pos(self.emb(idx), start_pos=start_pos)
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, mask=mask, kv_cache=layer_cache)
            new_caches.append(new_cache)
        logits = self.head(self.final_norm(x))
        return logits, new_caches

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int, temperature: float = 0.8, top_k: int = 10):
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 1. Prefill Phase: process entire prompt with causal mask
        seq_len = tokens.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool().unsqueeze(0).unsqueeze(1)
        logits, kv_caches = self(tokens, mask=causal_mask, kv_caches=None, start_pos=0)

        generated = list(prompt_ids)
        next_token_logits = logits[:, -1, :]

        # 2. Decode Phase: step 1 token at a time using KV Cache (no mask needed since query len is 1!)
        for step in range(max_new_tokens):
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated.append(token_id)
            if token_id == eos_token_id:
                break

            # Pass ONLY the newly sampled token + cached K/V
            logits, kv_caches = self(next_token, mask=None, kv_caches=kv_caches, start_pos=seq_len + step)
            next_token_logits = logits[:, -1, :]

        return generated