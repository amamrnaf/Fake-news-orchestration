"""Splitting, grouping, and leakage control (Sec 4.2).

Implements:
  * stratified train/val/test split by veracity (and domain, for AMTCele)
  * leave-one-domain-out / leave-one-group-out (events for PHEME, topics for COCO)
  * near-duplicate detection (normalised-text exact match + semantic cosine
    similarity) so duplicates/near-duplicates and same-event/thread items are
    always assigned to a single partition, never split across train/val/test.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from utils.types import NewsItem

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_duplicate_groups(
    items: Sequence[NewsItem],
    semantic_encoder=None,
    similarity_threshold: float = 0.95,
) -> List[str]:
    """Return a partition-key per item: existing `group` (event/thread/topic) if
    present, else a duplicate-cluster id, else the item's own id.

    Exact duplicates are detected via normalised-text equality (cheap, exact).
    Near-duplicates are detected via cosine similarity of a semantic encoder's
    embeddings when `semantic_encoder` is provided (must implement
    `.encode(list[str]) -> np.ndarray`); this step is skipped otherwise so the
    function stays usable without loading a model.
    """
    n = len(items)
    uf = UnionFind(n)

    norm_to_idx: Dict[str, int] = {}
    for i, item in enumerate(items):
        key = normalise_text(item.text)
        if key in norm_to_idx:
            uf.union(norm_to_idx[key], i)
        else:
            norm_to_idx[key] = i

    if semantic_encoder is not None and n > 1:
        embeddings = semantic_encoder.encode([it.text for it in items])
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        sims = embeddings @ embeddings.T
        ii, jj = np.where(sims >= similarity_threshold)
        for a, b in zip(ii, jj):
            if a < b:
                uf.union(int(a), int(b))

    cluster_root_to_key: Dict[int, str] = {}
    keys: List[str] = [None] * n  # type: ignore[list-item]
    for i, item in enumerate(items):
        if item.group:
            keys[i] = f"group::{item.group}"
    for i, item in enumerate(items):
        if keys[i] is not None:
            continue
        root = uf.find(i)
        if root not in cluster_root_to_key:
            cluster_root_to_key[root] = f"dup::{items[root].id}"
        keys[i] = cluster_root_to_key[root]
    return keys


def stratified_split(
    items: Sequence[NewsItem],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    stratify_by_domain: bool = True,
    semantic_encoder=None,
) -> List[NewsItem]:
    """Stratified train/val/test split by veracity (and domain if requested),
    keeping duplicate/near-duplicate/same-event clusters intact within one split.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    partition_keys = assign_duplicate_groups(items, semantic_encoder=semantic_encoder)

    # Aggregate to the partition-key level so a whole cluster moves together.
    key_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, k in enumerate(partition_keys):
        key_to_indices[k].append(i)

    def cluster_strat_label(idxs: List[int]) -> str:
        item = items[idxs[0]]
        return f"{item.domain}|{item.label}" if stratify_by_domain and item.domain else str(item.label)

    unique_keys = list(key_to_indices.keys())
    strat_labels = [cluster_strat_label(key_to_indices[k]) for k in unique_keys]

    # Drop strata with a single member from stratification (sklearn requires >=2 per class).
    label_counts = defaultdict(int)
    for l in strat_labels:
        label_counts[l] += 1
    safe_strat = [l if label_counts[l] >= 2 else "rare" for l in strat_labels]

    train_keys, rest_keys, train_lbls, rest_lbls = train_test_split(
        unique_keys, safe_strat, train_size=train_size, random_state=seed, stratify=safe_strat
    )
    rel_val = val_size / (val_size + test_size)
    rest_label_counts = defaultdict(int)
    for l in rest_lbls:
        rest_label_counts[l] += 1
    safe_rest = [l if rest_label_counts[l] >= 2 else "rare" for l in rest_lbls]
    val_keys, test_keys = train_test_split(
        rest_keys, train_size=rel_val, random_state=seed, stratify=safe_rest
    )

    split_of_key = {k: "train" for k in train_keys}
    split_of_key.update({k: "val" for k in val_keys})
    split_of_key.update({k: "test" for k in test_keys})

    out: List[NewsItem] = []
    for k, idxs in key_to_indices.items():
        split = split_of_key[k]
        for i in idxs:
            item = items[i]
            item.split = split
            out.append(item)
    return out


def leave_one_group_out(
    items: Sequence[NewsItem], held_out_group: str
) -> Dict[str, List[NewsItem]]:
    """Generic leave-one-domain/event/topic-out split.

    Uses `NewsItem.domain` when set (AMTCele leave-one-domain-out), otherwise
    `NewsItem.group` (PHEME leave-one-event-out, COCO held-out topic groups).
    """
    source, held_out = [], []
    for item in items:
        key = item.domain if item.domain else item.group
        (held_out if key == held_out_group else source).append(item)
    return {"source": source, "held_out": held_out}


def available_groups(items: Sequence[NewsItem]) -> List[str]:
    groups = {item.domain if item.domain else item.group for item in items}
    return sorted(g for g in groups if g)
