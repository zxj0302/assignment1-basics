from collections.abc import Iterable, Iterator
import regex as re
import json
from cs336_basics.config import * 


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_id = {(self.vocab_id[m[0]], self.vocab_id[m[1]]): i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens
        self.special_tokens_utf_8 = [s.encode("utf-8") for s in special_tokens] if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # NOTE: this is only for my own format
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(k): v.encode("latin-1")
            for k, v in raw_vocab.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = []
        for item in raw_merges:
            merges.append((item[0].encode("latin-1"), item[1].encode("latin-1")))

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def pretokenize(text, special_tokens):
        # NOTE: shouldn't concat pre_pat and PAT (pre_pat+'|'+PAT) for finditer or findall, as (front) part of the special tokens might be combined with chars adead (e.g., blank space) which leads to the special tokens broken and not matched correctly.
        pre_pat = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True))) if special_tokens else ""
        text = re.split(f"({pre_pat})", text) if special_tokens else [text]
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        result = [t.encode("utf-8") for i, t_spe in enumerate(text) for t in ([m.group() for m in re.finditer(PAT, t_spe)] if i % 2 == 0 else [t_spe])]
        return result

    def encode(self, text: str) -> list[int]:
        pre_token = self.pretokenize(text, self.special_tokens)
        # print(pre_token)
        result = []
        for pt in pre_token:
            if self.special_tokens_utf_8 and pt in self.special_tokens_utf_8:
                result += [self.vocab_id[pt]]
            else:
                pt = [self.vocab_id[bytes([b])] for b in pt]
                while True:
                    min_id = len(self.merges)
                    index = len(pt)
                    for i in range(len(pt)-1):
                        if (pt[i], pt[i+1]) in self.merges_id and self.merges_id[(pt[i], pt[i+1])] < min_id:
                            min_id = self.merges_id[(pt[i], pt[i+1])]
                            index = i

                    if min_id == len(self.merges):
                        break
                    else:
                        pt = pt[:index] + [self.vocab_id[self.vocab[pt[index]]+self.vocab[pt[index+1]]]] + pt[index+2:]
                result.extend(pt)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        result = b''.join([self.vocab[id] for id in ids])
        return result.decode("utf-8", errors='replace')
    

if __name__ == "__main__":
    vocab_filepath = vocab_ts_10k_filepath
    merges_filepath = merges_ts_10k_filepath

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # text = "They say it'll be a good idea🙃."
    # text = "🙃"
    # text = "你好！"
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    print(f"Text: {text}")
    encoded = tokenizer.encode(text)
    print(f"Encoded to: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded to: {decoded}")
    print("Correct!" if text == decoded else "Incorrect!!!")