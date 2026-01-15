"""
adapter.py run commands:
$env:PYTHONUTF8 = "1"
uv run pytest tests/test_tokenizer.py
"""
from __future__ import annotations
from collections.abc import Iterable, Iterator
import regex as re
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, 
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.invert_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []
        
        self.merge_order = {m: i for i, m in enumerate(self.merges)}
        self.special_order = sorted(self.special_tokens, key=len, reverse=True)
        self.special_hash = set(self.special_order)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json

        with open(vocab_filepath, encoding="utf-8") as fin:
            vocab = json.load(fin)
        
        with open(merges_filepath, encoding="utf-8") as fin:
            merges = json.load(fin)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for cur in iterable:
            text = cur.replace("\r\n", "\n").replace("\r", "\n")
            if self.special_order:
                # Remove all the special tokens before pretokenization
                splitter = re.compile("(" + "|".join(re.escape(token) for token in self.special_order) + ")")
                split_text = splitter.split(text)
            else:
                split_text = [text]

            for segment in split_text:
                if segment in self.special_hash:
                    yield self.invert_vocab[segment.encode("utf-8")]
                    continue
                for i in PAT.finditer(segment):
                    pretoken = i.group(0)

                    btext = [bytes([b]) for b in pretoken.encode("utf-8")]
                    # Either set up priority queue or quadratic solution

                    while True:
                        min_count = len(self.merge_order)
                        merge = None
                        for j in range(len(btext) - 1):
                            cur_merge = (btext[j], btext[j + 1])
                            cur_count = self.merge_order.get(cur_merge, min_count)
                            if min_count > cur_count:
                                min_count = cur_count
                                merge = cur_merge
                        if min_count == len(self.merge_order):
                            break

                        btemp = []
                        j = 0
                        while j < len(btext):
                            if (j < len(btext) - 1) and ((btext[j], btext[j + 1]) == merge):
                                btemp.append(btext[j] + btext[j + 1])
                                j += 2
                            else:
                                btemp.append(btext[j])
                                j += 1
                        btext = btemp
                
                    # newbtext = []
                    # for merge in self.merges:
                    #     for i in range(len(btext) - 1):
                    #         if merge == (btext[i], btext[i + 1]):
                    #             flag = True
                    #             newbtext.append(btext[i] + btext[i + 1])
                    #             i += 1
                    #         else:
                    #             flag = False
                    #             newbtext.append(btext[i])
                    #     if not flag:
                    #         newbtext.append(btext[-1])
                    #     btext = newbtext
                    #     newbtext = []
                    
                    for j in btext:
                        yield self.invert_vocab[j]
        
    def decode(self, ids: list[int]) -> str:
        cur = bytearray()
        for id in ids:
            cur.extend(self.vocab[id])
        
        return cur.decode("utf-8", errors="replace")