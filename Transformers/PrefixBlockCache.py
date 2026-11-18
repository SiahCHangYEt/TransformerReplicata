from dataclasses import dataclass
import torch

BLOCK_SIZE = 16

@dataclass
class KVBlock:
    # List of (K_block, V_block) tuples, one tuple per Transformer layer
    # Each tensor has shape: (1, num_heads, BLOCK_SIZE, d_k)
    layer_kvs: list[tuple[torch.Tensor, torch.Tensor]]
    ref_count: int = 0

def block_hashes(token_ids: list[int], salt=None) -> list[int]:
    """Chain-hash a token sequence into per-block keys."""
    hashes, parent = [], hash(salt)
    for start in range(0, len(token_ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = tuple(token_ids[start : start + BLOCK_SIZE])
        parent = hash((parent, block))
        hashes.append(parent)
    return hashes

class PrefixCacheManager:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.cache: dict[int, KVBlock] = {}

    def schedule(self, token_ids: list[int]):
        """Find longest cached prefix, increment ref_counts, and return tensors + remaining tokens."""
        hashes = block_hashes(token_ids)
        matched_hashes = []
        max_reusable_blocks = (len(token_ids) - 1) // BLOCK_SIZE

        for h in hashes[:max_reusable_blocks]:
            if h not in self.cache:
                break
            self.cache[h].ref_count += 1
            matched_hashes.append(h)

        reused_tokens = len(matched_hashes) * BLOCK_SIZE
        to_prefill = token_ids[reused_tokens:]

        # Stitch cached blocks back into per-layer (K, V) tensors for MultiHeadAttention
        if not matched_hashes:
            kv_caches = None
        else:
            kv_caches = []
            for layer_idx in range(self.num_layers):
                k_cat = torch.cat([self.cache[h].layer_kvs[layer_idx][0] for h in matched_hashes], dim=2)
                v_cat = torch.cat([self.cache[h].layer_kvs[layer_idx][1] for h in matched_hashes], dim=2)
                kv_caches.append((k_cat, v_cat))

        return matched_hashes, reused_tokens, to_prefill, kv_caches

    def store_completed_blocks(self, token_ids: list[int], full_layer_kvs: list[tuple[torch.Tensor, torch.Tensor]]):
        """Slice newly computed full K/V tensors into BLOCK_SIZE chunks and store any missing hashes."""
        hashes = block_hashes(token_ids)
        for block_idx, h in enumerate(hashes):
            if h not in self.cache:
                start = block_idx * BLOCK_SIZE
                end = start + BLOCK_SIZE
                layer_slices = [
                    (k[:, :, start:end, :].detach(), v[:, :, start:end, :].detach())
                    for (k, v) in full_layer_kvs
                ]
                self.cache[h] = KVBlock(layer_kvs=layer_slices, ref_count=1)


