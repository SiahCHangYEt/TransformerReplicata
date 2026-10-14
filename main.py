import torch
from Tokenization.BytePairTokenizer2 import BytePairTokenizer
from Embedding.Embeddings import TransformerEmbedding
from Transformers.PositionalEncoding import PositionalEncoding
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Generates a lower-triangular boolean mask: shape (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(1)


def run_full_pipeline_test():
    print("=== 1. TESTING BPE TOKENIZER ===")
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "the dog slept on the rug"
    ]
    tokenizer = BytePairTokenizer(vocab_size=60)
    tokenizer.train(corpus)
    print(f"[✓] Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")

    sample_text = "the cat sat"
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)
    print(f"Original text: '{sample_text}'")
    print(f"Encoded token IDs: {token_ids}")
    print(f"Decoded text: '{decoded_text}'")
    assert decoded_text == sample_text, "Tokenizer round-trip failed!"
    print("[✓] Tokenizer round-trip successful.")

    print("\n=== 2. TESTING ENCODER-DECODER ARCHITECTURE ===")
    device = torch.device("cpu")
    d_model = 64
    n_head = 4
    d_ff = 128
    num_layers = 2
    vocab_size = len(tokenizer.vocab)

    # Modules
    emb = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
    pos = PositionalEncoding(d_model=d_model)
    encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, h=n_head, d_ff=d_ff)
    decoder = TransformerDecoder(vocab_size=vocab_size, num_layers=num_layers, d_model=d_model, n_head=n_head, d_ff=d_ff)

    # Source: [bos, the, cat, eos, pad] (length = 5)
    src_tokens = torch.tensor([[2, 4, 5, 3, 0]], device=device)
    # Target: [bos, the, mat, eos] (length = 4)
    tgt_tokens = torch.tensor([[2, 4, 6, 3]], device=device)

    # Masks
    src_padding_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, 5)
    tgt_causal_mask = build_causal_mask(tgt_tokens.size(1), device)  # (1, 1, 4, 4)

    # Add this right before the assertion in main.py:
    print("Target Causal Mask:")
    print(tgt_causal_mask[0, 0])

    # Forward Encoder
    src_x = pos(emb(src_tokens))
    enc_out = encoder(src_x, mask=src_padding_mask)
    print(f"[✓] Encoder output shape: {enc_out.shape}")  # (1, 5, 64)

    # Forward Decoder
    tgt_x = pos(emb(tgt_tokens))
    logits = decoder(
        x=tgt_x,
        enc_output=enc_out,
        tgt_mask=tgt_causal_mask,
        memory_mask=src_padding_mask
    )
    print(f"[✓] Decoder logits shape: {logits.shape}")  # (1, 4, vocab_size)
    assert logits.shape == (1, 4, vocab_size), "Logits shape mismatch!"

    print("\n=== 3. VERIFYING CAUSAL AUTOREGRESSIVE INVARIANCE ===")

    # 1. Turn off dropout globally
    emb.eval()
    if hasattr(pos, "eval"):
        pos.eval()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Pass 1
        src_x = pos(emb(src_tokens))
        enc_out = encoder(src_x, mask=src_padding_mask)

        tgt_x = pos(emb(tgt_tokens))
        logits = decoder(tgt_x, enc_out, tgt_mask=tgt_causal_mask, memory_mask=src_padding_mask)

        # Pass 2: Mutate future token
        tgt_tokens_perturbed = tgt_tokens.clone()
        tgt_tokens_perturbed[0, 2] = 8

        tgt_x_perturbed = pos(emb(tgt_tokens_perturbed))
        logits_perturbed = decoder(tgt_x_perturbed, enc_out, tgt_mask=tgt_causal_mask, memory_mask=src_padding_mask)

        # Difference at position 0 (should be strictly 0.0)
        diff_pos0 = torch.abs(logits[:, 0, :] - logits_perturbed[:, 0, :]).max().item()
        print(f"Difference strictly at Position 0: {diff_pos0:.6e}")

        diff = torch.abs(logits[:, :2, :] - logits_perturbed[:, :2, :]).max().item()
        print(f"Max difference at past positions (0 and 1): {diff:.6e}")
        assert diff < 1e-5, f"Causal mask leaked future information! Diff: {diff}"




if __name__ == "__main__":
    run_full_pipeline_test()
