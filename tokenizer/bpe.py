"""
Tokenizer is responsible for converting a given string into a set of random id (and vice versa) that can be fed to the model.
The first step of training a LLM would be to train the tokenizer. This is generally done on the same text as that used to
train the model. 

This file implements Byte Pair Encoding (BPE) algorithm which is the most common tokenizer used in LLMs. It only implments tokenization
english language, which is why we assume an intial vocab length of 256 (ASCII set list)
"""
import unicodedata
from typing import Sequence, Mapping, List
from itertools import pairwise

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer():
    def __init__(self):
        self.merges = {}
        # id to byte string
        self.vocab = {}
        # byte string to id
        self.revese_vocab_mapping = {}
        
    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for merge_pair in self.merges:
            vocab[self.merges[merge_pair]] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        return vocab

    def _get_stats(self, ids: Sequence[int]) -> Mapping[tuple[int], int]:
        counts = {}
        for pair in pairwise(ids):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pairs(self, ids: Sequence[int], pair: tuple[int], merge_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                new_ids.append(merge_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
        
    def train(self, text: str, vocab_size: int) -> None:
        """
        trains the tokenizer on a given text
        Args:
        - text: text on which tokenizer will be trained.
        - vocab_size: Vocabulary size. This will determine the token length when encoding strings, the larger the vocabulary size the shorter the tokenized string will be.
        """
        assert vocab_size >= 256
        # Beyond 256 character the remaining vocab will be based on merged characters.
        num_merges = vocab_size - 256

        self.vocab = {i: bytes([i]) for i in range(256)}
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes)
        
        for merge_count in range(num_merges):
            pair_stats = self._get_stats(ids)
            top_pair = max(pair_stats, key=lambda x: pair_stats.get(x))
            new_id = 256 + merge_count
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            ids = self._merge_pairs(ids, top_pair, new_id)

    def encode(self, text:str) -> Sequence[int]:
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes)
        encoded_list = []
        i = 0
        while i < len(ids):
            curr_id = ids[i]
            while i < len(ids) - 1:
                # check if the pair exists in merges
                next_id = ids[i+1]
                if self.merges.get((curr_id, next_id), None) is None:
                   break
                curr_id = self.merges[(curr_id, next_id)]
                i += 1
            encoded_list.append(curr_id)
            i += 1
        return encoded_list
                

    def decode(self, ids: Sequence[int]) -> str:
        decoded_str = ""
        for id in ids:
            decoded_str += self.vocab[id].decode("utf-8")
        return decoded_str

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only

        Based off - https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("bpe v1\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()