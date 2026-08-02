"""T1: Supervised transformer veracity detector (Table 1).

'Fine-tuned encoder trained on permitted source-domain examples. Fast and
strong for patterns represented in training data. Primary cost: one encoder
forward pass.'
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from detectors.base import Detector
from eval.metrics import macro_f1
from utils.types import DetectorOutput, NewsItem


@dataclass
class T1Config:
    model_name: str = "roberta-base"
    lr: float = 2e-5
    batch_size: int = 16
    max_length: int = 256
    epochs: int = 10
    patience: int = 3
    weight_decay: float = 0.01
    seed: int = 42


class SupervisedTransformerDetector(Detector):
    tool_id = "T1"
    description = "Fine-tuned encoder trained on permitted source-domain examples."
    expected_strength = "Fast and strong for patterns represented in training data."
    primary_cost = "One encoder forward pass."

    def __init__(self, cfg: T1Config, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2
        ).to(self.device)

    def fit(self, train_items: Sequence[NewsItem], val_items: Sequence[NewsItem]) -> float:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset

        from utils.seed import set_seed

        set_seed(self.cfg.seed)

        class _Ds(Dataset):
            def __init__(self, items):
                self.items = list(items)

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                it = self.items[idx]
                return it.text, it.label

        def collate(batch):
            texts, labels = zip(*batch)
            return list(texts), torch.tensor(labels, dtype=torch.long)

        loader = DataLoader(_Ds(train_items), batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate)
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best_f1, best_state, no_improve = -1.0, None, 0
        for _ in range(self.cfg.epochs):
            self.model.train()
            for texts, labels in loader:
                optimizer.zero_grad()
                enc = self._tokenize(texts)
                out = self.model(**enc, labels=labels.to(self.device))
                out.loss.backward()
                optimizer.step()

            val_f1 = self._eval_macro_f1(val_items)
            if val_f1 > best_f1:
                best_f1, best_state, no_improve = val_f1, copy.deepcopy(self.model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_f1

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt"
        ).to(self.device)

    def _eval_macro_f1(self, items: Sequence[NewsItem]) -> float:
        preds = [self.predict(it).label for it in items]
        gold = [it.label for it in items]
        return macro_f1(gold, preds)

    def predict(self, item: NewsItem) -> DetectorOutput:
        import torch

        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            enc = self._tokenize([item.text])
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        latency = time.perf_counter() - start
        label = int(np.argmax(probs))
        return DetectorOutput(
            tool_id=self.tool_id,
            label=label,
            prob_fake=float(probs[1]),
            latency_s=latency,
            generated_tokens=0,
        )
