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


def load_jsonl_cache(cache_path: Optional[str]) -> dict:
    """{item_id: output} from a write-through JSONL cache. Skips lines that
    don't parse: a power-off mid-write leaves a truncated last line, and the
    item on it is simply recomputed on the next run."""
    cached: dict = {}
    if not cache_path or not os.path.exists(cache_path):
        return cached
    skipped = 0
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cached[row["item_id"]] = row["output"]
            except (json.JSONDecodeError, KeyError):
                skipped += 1
    if skipped:
        print(f"[cache] {cache_path}: skipped {skipped} unreadable line(s); those items will be recomputed", flush=True)
    return cached


def open_jsonl_for_append(cache_path: str):
    """Opens a JSONL cache for appending, first terminating a truncated last
    line so the next record isn't glued onto it."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write("\n")
    return open(cache_path, "a", encoding="utf-8")


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
    cached = load_jsonl_cache(cache_path)

    total = len(item_ids)
    start = time.perf_counter()
    results: List[T] = []
    cache_file = None
    if cache_path:
        cache_file = open_jsonl_for_append(cache_path)
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
