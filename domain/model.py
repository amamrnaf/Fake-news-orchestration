"""Domain classifier model + input construction for the 3 variants in Sec 3.3:
  (i)   target text only
  (ii)  target text + semantic neighbours
  (iii) target text + affective-semantic neighbours

p(d|x) = softmax(W h_x + b) over the 7 AMTCele domains.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

from retrieval.index import RetrievedExample
from utils.types import DOMAINS

NEIGHBOUR_SEP = " [NEIGHBOUR] "


def build_domain_input_text(
    text: str,
    neighbours: Optional[Sequence[RetrievedExample]] = None,
    max_neighbours: int = 4,
) -> str:
    """Concatenates the target text with retrieved-neighbour snippets.

    `neighbours=None` reproduces variant (i). Passing semantic-only-retrieved
    neighbours (alpha=1) reproduces variant (ii); affective-semantic neighbours
    (tuned alpha) reproduce variant (iii). The retrieval alpha therefore
    controls which variant this function realises — callers select alpha
    upstream when building `neighbours`.
    """
    if not neighbours:
        return text
    snippets = [n.item.text[:200] for n in neighbours[:max_neighbours]]
    return text + NEIGHBOUR_SEP + NEIGHBOUR_SEP.join(snippets)


class DomainClassifierModel:
    """Thin wrapper around a HF encoder + sequence classification head."""

    def __init__(self, model_name: str, num_labels: int = len(DOMAINS), max_length: int = 256, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Explicit fp32: without this, some transformers versions resolve a
        # checkpoint's dtype from hub metadata on CUDA, and DeBERTa-v3 trained
        # in fp16 produces NaN almost immediately (AdamW's default eps=1e-8
        # underflows fp16 precision). Every one of 12 configs for
        # microsoft/deberta-v3-base diverged to NaN from epoch 1 on Colab's
        # T4 with no explicit dtype; gradient clipping alone didn't help,
        # consistent with the NaN originating in the forward pass, not
        # exploding-but-finite gradients.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, torch_dtype=torch.float32
        ).to(self.device)
        self.id2label = {i: d for i, d in enumerate(DOMAINS)}
        self.label2id = {d: i for i, d in enumerate(DOMAINS)}

    def tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "wrapper_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"max_length": self.max_length}, f)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DomainClassifierModel":
        with open(os.path.join(path, "wrapper_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return cls(model_name=path, max_length=meta["max_length"], device=device)
