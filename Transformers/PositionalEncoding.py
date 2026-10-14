import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1)]
        return x

    def  print(self):
        print(torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512)))


if __name__ == "__main__":
    pe = PositionalEncoding(512,5000)
    pe.print()


