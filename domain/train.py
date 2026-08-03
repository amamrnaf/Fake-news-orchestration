"""Training loop for the domain classifier with a hyperparameter grid and
early stopping on validation macro-F1 (Sec 4.4):
  lr in {1e-5, 2e-5, 3e-5}, batch_size in {8, 16}, max_seq_len in {256, 512}.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence

import numpy as np

from domain.model import DomainClassifierModel, build_domain_input_text
from eval.metrics import macro_f1
from utils.types import DOMAINS, NewsItem

LR_GRID = [1e-5, 2e-5, 3e-5]
BATCH_SIZE_GRID = [8, 16]
MAX_LEN_GRID = [256, 512]


@dataclass
class TrainConfig:
    model_name: str
    lr: float
    batch_size: int
    max_length: int
    epochs: int = 10
    patience: int = 3
    weight_decay: float = 0.01
    seed: int = 42


def _labels_for(items: Sequence[NewsItem]) -> np.ndarray:
    label2id = {d: i for i, d in enumerate(DOMAINS)}
    return np.array([label2id[it.domain] for it in items])


def train_one_config(
    train_items: Sequence[NewsItem],
    val_items: Sequence[NewsItem],
    cfg: TrainConfig,
    train_neighbours: Optional[List] = None,
    val_neighbours: Optional[List] = None,
) -> tuple[DomainClassifierModel, float]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset

    from utils.seed import set_seed

    set_seed(cfg.seed)

    clf = DomainClassifierModel(cfg.model_name, max_length=cfg.max_length)

    train_texts = [
        build_domain_input_text(it.text, train_neighbours[i] if train_neighbours else None)
        for i, it in enumerate(train_items)
    ]
    val_texts = [
        build_domain_input_text(it.text, val_neighbours[i] if val_neighbours else None)
        for i, it in enumerate(val_items)
    ]
    train_labels = _labels_for(train_items)
    val_labels = _labels_for(val_items)

    class _TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts, self.labels = texts, labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], int(self.labels[idx])

    def collate(batch):
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(
        _TextDataset(train_texts, train_labels), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )

    optimizer = AdamW(clf.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    train_start = time.perf_counter()
    for epoch in range(cfg.epochs):
        clf.model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            enc = clf.tokenize(list(texts))
            out = clf.model(**enc, labels=labels.to(clf.device))
            out.loss.backward()
            optimizer.step()

        val_f1 = evaluate_macro_f1(clf, val_texts, val_labels)
        elapsed = time.perf_counter() - train_start
        print(
            f"  [{cfg.model_name}] epoch {epoch + 1}/{cfg.epochs} | val_f1={val_f1:.3f} | "
            f"best_f1={max(val_f1, best_f1):.3f} | elapsed={elapsed:.0f}s",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(clf.model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"  [{cfg.model_name}] early stopping at epoch {epoch + 1} (patience={cfg.patience})", flush=True)
                break

    if best_state is not None:
        clf.model.load_state_dict(best_state)
    return clf, best_f1


def evaluate_macro_f1(clf: DomainClassifierModel, texts: List[str], labels: np.ndarray, batch_size: int = 32) -> float:
    import torch

    clf.model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = clf.tokenize(batch)
            logits = clf.model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return macro_f1(labels.tolist(), preds)


def grid_search_domain_classifier(
    model_names: Sequence[str],
    train_items: Sequence[NewsItem],
    val_items: Sequence[NewsItem],
    train_neighbours: Optional[List] = None,
    val_neighbours: Optional[List] = None,
    lr_grid: Sequence[float] = LR_GRID,
    batch_size_grid: Sequence[int] = BATCH_SIZE_GRID,
    max_len_grid: Sequence[int] = MAX_LEN_GRID,
    seed: int = 42,
) -> tuple[DomainClassifierModel, TrainConfig, float]:
    """Selects the principal classifier by validation macro-F1 (Sec 3.3),
    not test accuracy, across candidate encoders and hyperparameters."""
    best_model, best_cfg, best_f1 = None, None, -1.0
    combos = list(product(model_names, lr_grid, batch_size_grid, max_len_grid))
    grid_start = time.perf_counter()
    for i, (model_name, lr, bs, max_len) in enumerate(combos):
        cfg = TrainConfig(model_name=model_name, lr=lr, batch_size=bs, max_length=max_len, seed=seed)
        print(
            f"[grid search] config {i + 1}/{len(combos)}: {model_name} lr={lr} batch_size={bs} max_length={max_len}",
            flush=True,
        )
        model, f1 = train_one_config(train_items, val_items, cfg, train_neighbours, val_neighbours)
        elapsed = time.perf_counter() - grid_start
        print(
            f"[grid search] config {i + 1}/{len(combos)} finished: val_f1={f1:.3f} | "
            f"total_elapsed={elapsed:.0f}s",
            flush=True,
        )
        if f1 > best_f1:
            best_model, best_cfg, best_f1 = model, cfg, f1
    return best_model, best_cfg, best_f1
