import torch
from torch.utils.data import Dataset
from Tokenization.BytePairTokenizer2 import BytePairTokenizer

class TextTokenDataset(Dataset):
    def __init__(self, file_path:str, tokenizer:BytePairTokenizer, seq_len = 64):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8" ) as f:
            text = f.read()
        token_ids = tokenizer.encode(text,add_special_tokens=False)
        self.data = torch.tensor(token_ids,dtype = torch.long)
        if len(self.data) <= self.seq_len:
            raise ValueError(
                f"Text file only produced {len(self.data)} tokens, "
                f"which is less than seq_len ({self.seq_len} + 1). Provide more text."
            )

    def __len__(self) -> int:
        # Number of non-overlapping (or strided) chunks we can extract
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len

        # Input: tokens [start : end]
        # Target: tokens [start+1 : end+1] (shifted by 1 for next-token prediction)
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        return x, y


if __name__ == "__main__":
    from Tokenization.BytePairTokenizer2 import BytePairTokenizer

    tok = BytePairTokenizer(vocab_size=200)
    tok.train(["hello world from transformer", "the quick brown fox jumps over the lazy dog"])

    # Quick dummy write to verify logic
    with open("sample.txt", "w") as f:
        f.write("hello world from transformer the quick brown fox jumps over the lazy dog " * 10)

    ds = TextTokenDataset("sample.txt", tok, seq_len=16)
    x, y = ds[0]
    print("Sample input shape :", x.shape)
    print("Sample target shape:", y.shape)
    print("Input tokens       :", x[:6].tolist())
    print("Target tokens (+1) :", y[:6].tolist())