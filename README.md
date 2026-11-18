# TransformerReplicata

A modular, from-scratch implementation of the Transformer architecture in Python, featuring custom Byte-Pair Encoding (BPE) tokenization and block-level KV prefix caching for efficient autoregressive inference.

---

## Key Features

* **From-Scratch Transformer Architecture:** Implements scaled dot-product multi-head attention, positional encoding, position-wise feed-forward networks, and encoder-decoder blocks without relying on high-level `nn.Transformer` wrappers.
* **Custom BPE Subword Tokenization:** Includes an iterative Byte-Pair Encoding (BPE) tokenizer built from scratch to construct subword vocabularies and handle sequence encoding/decoding.
* **Block-Level Prefix Caching (`PrefixBlockCache`):** Optimizes inference memory and compute by partitioning key-value (KV) caches into fixed-size blocks (`BLOCK_SIZE = 16`) and hashing block sequences to reuse KV states across shared prompt prefixes.
* **End-to-End Modular Pipeline:** Clean separation of data loading, tokenization, model layers, training loops, and inference execution.

---

## Project Structure

```text
TransformerReplicata/
├── Embedding/
│   └── Embeddings.py            # Token embedding layer
├── Tokenization/
│   ├── BytePairTokenizer.py     # Initial BPE tokenizer implementation
│   └── BytePairTokenizer2.py    # Refined BPE subword tokenizer
├── Transformers/
│   ├── FeedForward.py           # Position-wise feed-forward network
│   ├── MultiHeadAttention.py    # Scaled dot-product multi-head attention
│   ├── PositionalEncoding.py    # Sequence positional encodings
│   └── PrefixBlockCache.py      # Block-hashed KV prefix cache for inference
├── Dataset.py                   # Custom dataset and batching utilities
├── Decoder.py                   # Transformer decoder stack
├── Encoder.py                   # Transformer encoder stack
├── Pipeline.py                  # End-to-end training and inference orchestration
├── Training.py                  # Training loop and optimization logic
└── main.py                      # Entry point for running training and generation

Getting Started
Prerequisites
Python 3.9+

PyTorch / NumPy (update with exact dependencies in requirements.txt)
