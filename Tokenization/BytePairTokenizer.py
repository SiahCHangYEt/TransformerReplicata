import regex as re
from collections import Counter
class BytePairTokenizer:
    def __init__(self):
        self.merges = {}
        self.gpt_2_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def get_freq_from_vocab(self,vocab_dict:dict)->dict:
        stats = {}
        for word_tuple, count in vocab_dict.items():
            for pair in zip(word_tuple, word_tuple[1:]):
                stats[pair] = stats.get(pair, 0) + count
        return stats

    def merge_vocab(self,vocab_dict:dict,pair:tuple,idx:int)->dict:
        new_vocab = {}
        for word_tuple, count in vocab_dict.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[word_tuple] = count
        return new_vocab

    def regex_split(self,text:str)->list:
        return self.gpt_2_pattern.findall(text)

    def encode(self,text:str)->list[int]:
        chunks = self.regex_split(text)
        ids_per_chunk = [chunk.encode("utf-8") for chunk in chunks]
        for pair, idx in self.merges.items():
            ids_per_chunk = [self.merge_list(chunk_ids,pair,idx) for chunk_ids in ids_per_chunk]
        return [idx for chunk in ids_per_chunk for idx in chunk]

    def merge_list(self,ids:list,pair:tuple,idx:int)->list:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) and pair[0] == ids[i] and pair[1] == ids[i]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def tokenize(self,text:str,vocab_size:int):
        chunks = self.regex_split(text)
        vocab_dict = Counter(tuple(chunk.encode("utf-8")) for chunk in chunks)
        num_merges = vocab_size - 256
        for i in range (num_merges):
            stats = self.get_freq_from_vocab(vocab_dict)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            vocab_dict = self.merge_vocab(vocab_dict, pair, idx)
            self.merges[pair] = idx
        



