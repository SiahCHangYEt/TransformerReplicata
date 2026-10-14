import torch
import torch.nn as nn
import math

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        # The learnable weight matrix W_e of shape (vocab_size, d_model)
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token IDs of shape (batch_size, seq_len)
        Returns:
            Tensor of scaled embeddings of shape (batch_size, seq_len, d_model)
        """
        # Multiply by sqrt(d_model) as specified in Vaswani et al.
        return self.lut(x) * math.sqrt(self.d_model)