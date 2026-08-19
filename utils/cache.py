"""Generic per-item resumable caching + progress logging, for loops over
items where each item's result should survive a crash and be skippable on
resume. Same idea as detectors.base.predict_batch_with_cache, generalized
for loops that aren't a plain Detector.predict(item) call -- e.g.
LLMRouter.select(), which needs several per-item context args (domain
prediction, affective summary, retrieved examples) that a single-item
predict() interface can't express.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


def cached_map(
    item_ids: List[str],
    compute: Callable[[int], T],
    from_dict: Callable[[dict], T],
    cache_path: Optional[str] = None,
    label: str = "item",
) -> List[T]:
    """Runs compute(i) for each index into item_ids, in order.

    If `cache_path` is given: results already present there (keyed by
    item_id) are loaded via `from_dict` instead of recomputed, and each
    freshly computed result is appended to the file immediately (not
    batched to the end) via `dataclasses.asdict`, so a crash only loses
    whatever wasn't cached yet. Caller is responsible for making sure
    `cache_path` is scoped correctly for what's being cached -- e.g. LODO's
    per-fold cache dirs, since a router decision from one fold's retrieval
    index would be wrong to reuse in another fold.
    """
    cached: dict = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cached[row["item_id"]] = row["output"]

    total = len(item_ids)
    start = time.perf_counter()
    results: List[T] = []
    cache_file = None
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cache_file = open(cache_path, "a", encoding="utf-8")
    try:
        for i, item_id in enumerate(item_ids):
            if item_id in cached:
                out = from_dict(cached[item_id])
            else:
                out = compute(i)
                if cache_file:
                    out_dict = asdict(out) if is_dataclass(out) else out
                    cache_file.write(json.dumps({"item_id": item_id, "output": out_dict}) + "\n")
                    cache_file.flush()
            results.append(out)
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (total - i - 1) / rate if rate > 0 else float("nan")
            tag = " (cached)" if item_ids[i] in cached else ""
            print(f"[{label}] {i + 1}/{total} done{tag} | elapsed={elapsed:.0f}s | eta={eta:.0f}s", flush=True)
    finally:
        if cache_file:
            cache_file.close()
    return results
