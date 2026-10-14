from collections import Counter
import re


class BytePairTokenizer:
    def __init__(self, vocab_size: int = 50):
        self.target_vocab_size = vocab_size
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []

    def _get_stats(self, word_freqs: dict) -> Counter:
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        p0, p1 = pair
        pattern = re.compile(r'(?<!\S)' + re.escape(p0) + r'\s+' + re.escape(p1) + r'(?!\S)')
        replacement = p0 + p1
        return {
            pattern.sub(replacement, word): freq
            for word, freq in word_freqs.items()
        }

    def train(self, corpus: list[str]):
        # Base vocabulary initializes with special tokens
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}

        # Split corpus into character sequences with </w> indicating word ends
        words = []
        for line in corpus:
            for word in line.strip().split():
                words.append(" ".join(list(word)) + " </w>")
        word_freqs = Counter(words)

        # Collect unique characters into vocab
        for word in word_freqs:
            for ch in word.split():
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        # Iteratively merge most frequent adjacent pairs
        while len(self.vocab) < self.target_vocab_size:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_token = "".join(best_pair)
            self.vocab[merged_token] = len(self.vocab)
            self.merges.append(best_pair)

        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for word in text.strip().split():
            symbols = list(word) + ["</w>"]
            for p0, p1 in self.merges:
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                        new_symbols.append(p0 + p1)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            for sym in symbols:
                tokens.append(self.vocab.get(sym, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        words = []
        current_word = []
        for tid in token_ids:
            if tid in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            token_str = self.inverse_vocab.get(tid, "<unk>")
            if token_str.endswith("</w>"):
                current_word.append(token_str[:-4])
                words.append("".join(current_word))
                current_word = []
            else:
                current_word.append(token_str)
        if current_word:
            words.append("".join(current_word))
        return " ".join(words)