import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Tokenization.BytePairTokenizer2 import BytePairTokenizer
from Pipeline import MiniGPT
from Dataset import TextTokenDataset

# 1. Hardware Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Hyperparameters
vocab_size = 300  # Keep small for quick convergence on tiny texts
seq_len = 32  # Small context window for CPU-friendly training
batch_size = 16
num_epochs = 30
lr = 1e-3

# 3. Prepare Tokenizer & Dataset
data_path = "input.txt"

# Read sample corpus to train tokenizer
with open(data_path, "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f.readlines() if line.strip()]

tokenizer = BytePairTokenizer(vocab_size=vocab_size)
tokenizer.train(raw_lines[:500])  # Train on first 500 lines for fast startup

actual_vocab_size = len(tokenizer.vocab)
print(f"Trained vocabulary size: {actual_vocab_size}")

dataset = TextTokenDataset(file_path=data_path, tokenizer=tokenizer, seq_len=seq_len)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Model Setup
model = MiniGPT(
    vocab_size=actual_vocab_size,
    d_model=64,
    n_head=4,
    d_ff=128,
    num_layers=2,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def get_causal_mask(size: int, dev: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((size, size), device=dev)).bool().unsqueeze(0).unsqueeze(1)


# 5. Training Loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mask = get_causal_mask(inputs.size(1), device)

        optimizer.zero_grad()
        logits, _ = model(inputs, mask=mask)

        # CrossEntropyLoss expects (Batch * SeqLen, VocabSize) vs (Batch * SeqLen)
        loss = criterion(logits.view(-1, actual_vocab_size), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f}")

# 6. Sample Autoregressive Generation
model.eval()
prompt = "First Citizen:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

if prompt_tokens:
    generated_ids = model.generate(
        prompt_ids=prompt_tokens,
        max_new_tokens=25,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_k=5
    )
    print("\n--- Model Output ---")
    print(tokenizer.decode(generated_ids))