import json
import re
from collections import Counter
from pathlib import Path


class SimpleTextTokenizer:
    """
    Lightweight tokenizer for standalone keyframe scorer.

    It avoids AutoTokenizer/LlamaTokenizer and only provides:
      - pad_token_id
      - __len__()
      - __call__(...)
    """

    def __init__(self, vocab, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"):
        self.vocab = vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]

    @staticmethod
    def _tokenize(text):
        text = str(text).lower()
        return re.findall(r"[a-zA-Z0-9]+|[^\s\w]", text)

    @classmethod
    def from_annotation(cls, annotation_path, max_vocab_size=20000, min_freq=1):
        annotation_path = Path(annotation_path)
        data = json.load(open(annotation_path, "r", encoding="utf-8"))

        counter = Counter()
        for item in data:
            text = item.get("gpt_instruction", "")
            counter.update(cls._tokenize(text))

        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }

        for tok, freq in counter.most_common(max_vocab_size - len(vocab)):
            if freq < min_freq:
                continue
            if tok not in vocab:
                vocab[tok] = len(vocab)

        return cls(vocab)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.vocab, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        vocab = json.load(open(path, "r", encoding="utf-8"))
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding=False,
        return_attention_mask=True,
        **kwargs,
    ):
        tokens = self._tokenize(text)
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation and max_length is not None:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }
