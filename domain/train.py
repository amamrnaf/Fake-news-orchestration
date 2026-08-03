"""Training loop for the domain classifier with a hyperparameter grid and
early stopping on validation macro-F1 (Sec 4.4):
  lr in {1e-5, 2e-5, 3e-5}, batch_size in {8, 16}, max_seq_len in {256, 512}.
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import List, Optional, Sequence

import numpy as np

from domain.model import DomainClassifierModel, build_domain_input_text
from eval.metrics import macro_f1
from utils.types import DOMAINS, NewsItem


def _free_gpu_model(model: Optional[DomainClassifierModel]) -> None:
    """Releases a discarded config's GPU memory. Without this, a grid search
    over N hyperparameter combos leaves N models resident on the GPU at once
    (each iteration creates a fresh model but never frees the previous one),
    which is exactly what OOM'd Colab partway through xlm-roberta-base after
    roberta-base's and deberta-v3's 12-config searches had already piled up
    unreleased memory."""
    if model is None:
        return
    import torch

    model.model.to("cpu")
    del model.model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        for step, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            enc = clf.tokenize(list(texts))
            out = clf.model(**enc, labels=labels.to(clf.device))
            if not torch.isfinite(out.loss):
                # Fail loudly, immediately -- not after epochs of silently
                # "training" on NaN. This is exactly how microsoft/deberta-v3-base
                # burned ~1hr of Colab GPU time across 12 configs before anyone
                # noticed: loss was NaN from batch 1, every config, but nothing
                # checked for it until the final checkpoint was already saved.
                raise FloatingPointError(
                    f"[{cfg.model_name}] non-finite loss ({out.loss.item()}) at epoch {epoch + 1}, "
                    f"step {step + 1} -- stopping immediately rather than continuing on garbage. "
                    "Common causes: fp16/bf16 auto-loading on CUDA (see torch_dtype in DomainClassifierModel), "
                    "learning rate too high, or a numerically unstable architecture (DeBERTa-v3's disentangled "
                    "attention is a known case)."
                )
            out.loss.backward()
            # Defensive: bounds gradient norm in case of a genuine explosion
            # (as opposed to the immediate-NaN case above, which this alone
            # does not fix -- see the fp32 dtype fix in DomainClassifierModel).
            torch.nn.utils.clip_grad_norm_(clf.model.parameters(), max_norm=1.0)
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
            # Keep the snapshot on CPU, not GPU -- otherwise every improving
            # epoch holds a second full copy of the model in VRAM alongside
            # the live training copy, needlessly doubling peak GPU usage.
            best_state = {k: v.cpu() for k, v in clf.model.state_dict().items()}
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


def _checkpoint_path_for(checkpoint_dir: str, model_name: str) -> str:
    return os.path.join(checkpoint_dir, model_name.replace("/", "__"))


def load_domain_classifier_checkpoint(
    checkpoint_dir: str, model_name: str
) -> Optional[tuple[DomainClassifierModel, TrainConfig, float]]:
    """Loads a checkpoint saved by `grid_search_domain_classifier(..., checkpoint_dir=...)`
    (e.g. produced on Colab), or returns None if no checkpoint exists for this encoder yet."""
    path = _checkpoint_path_for(checkpoint_dir, model_name)
    meta_path = os.path.join(path, "grid_search_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    model = DomainClassifierModel.load(path)
    cfg = TrainConfig(**meta["train_cfg"])
    return model, cfg, meta["best_f1"]


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
    checkpoint_dir: Optional[str] = None,
) -> tuple[DomainClassifierModel, TrainConfig, float]:
    """Selects the principal classifier by validation macro-F1 (Sec 3.3),
    not test accuracy, across candidate encoders and hyperparameters.

    If `checkpoint_dir` is given and already has a saved checkpoint for this
    encoder (e.g. trained on Colab and copied back), that's loaded and the
    entire grid search is skipped. Otherwise the grid search runs as usual
    and, if `checkpoint_dir` is given, the winning model is saved there.
    Intended to be called once per encoder (as `run_domain_classification.py`
    does), so `checkpoint_dir` here is keyed by encoder name, not by combo.
    """
    if checkpoint_dir is not None and len(model_names) == 1:
        cached = load_domain_classifier_checkpoint(checkpoint_dir, model_names[0])
        if cached is not None:
            print(f"[grid search] loaded cached checkpoint for {model_names[0]}, skipping training", flush=True)
            return cached

    best_model, best_cfg, best_f1 = None, None, -1.0
    combos = list(product(model_names, lr_grid, batch_size_grid, max_len_grid))
    grid_start = time.perf_counter()
    for i, (model_name, lr, bs, max_len) in enumerate(combos):
        cfg = TrainConfig(model_name=model_name, lr=lr, batch_size=bs, max_length=max_len, seed=seed)
        print(
            f"[grid search] config {i + 1}/{len(combos)}: {model_name} lr={lr} batch_size={bs} max_length={max_len}",
            flush=True,
        )
        try:
            model, f1 = train_one_config(train_items, val_items, cfg, train_neighbours, val_neighbours)
        except FloatingPointError as exc:
            # Don't let one broken config (or a systematically unstable
            # encoder, as microsoft/deberta-v3-base turned out to be) abort
            # the whole grid search -- other encoders after this one in
            # candidate_encoders still deserve their turn. Skip and move on.
            elapsed = time.perf_counter() - grid_start
            print(
                f"[grid search] config {i + 1}/{len(combos)} FAILED ({exc}) | total_elapsed={elapsed:.0f}s -- skipping",
                flush=True,
            )
            continue
        elapsed = time.perf_counter() - grid_start
        print(
            f"[grid search] config {i + 1}/{len(combos)} finished: val_f1={f1:.3f} | "
            f"total_elapsed={elapsed:.0f}s",
            flush=True,
        )
        if f1 > best_f1:
            _free_gpu_model(best_model)  # this config beat the previous best -- release it
            best_model, best_cfg, best_f1 = model, cfg, f1
        else:
            _free_gpu_model(model)  # not an improvement -- release it, don't let it pile up in VRAM

    if best_model is None:
        raise RuntimeError(
            f"[grid search] every config failed for {model_names} -- nothing to save. "
            "Check the FAILED lines above for the underlying error."
        )

    if checkpoint_dir is not None and len(model_names) == 1:
        path = _checkpoint_path_for(checkpoint_dir, model_names[0])
        best_model.save(path)
        with open(os.path.join(path, "grid_search_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"train_cfg": asdict(best_cfg), "best_f1": best_f1}, f, indent=2)
        print(f"[grid search] saved checkpoint for {model_names[0]} to {path}", flush=True)

    return best_model, best_cfg, best_f1
