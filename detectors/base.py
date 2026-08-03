"""Common interface for every tool in the registry E = {T1, T2, T3} (Sec 3.4, Table 1)."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Optional

from utils.types import DetectorOutput, NewsItem


def predict_batch_with_cache(detector, items: List[NewsItem], cache_path: Optional[str] = None) -> List[DetectorOutput]:
    """Shared predict_batch implementation for any object with `.tool_id` and
    `.predict(item)`. `cache_path`, if given, makes this resumable: every
    prediction is appended to that JSONL file the moment it's computed (not
    batched to the end), and on the next run, items already present there are
    loaded instead of re-predicted. Without this, a run killed partway
    through (crash, session teardown, etc.) loses every prediction computed
    so far -- expensive to lose for LLM-backed tools (T2/T3), where each
    prediction can cost several seconds to tens of seconds."""
    cached: dict = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cached[row["item_id"]] = row["output"]

    total = len(items)
    start = time.perf_counter()
    outputs = []
    cache_file = None
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cache_file = open(cache_path, "a", encoding="utf-8")
    try:
        for i, item in enumerate(items):
            if item.id in cached:
                out = DetectorOutput(**cached[item.id])
            else:
                out = detector.predict(item)
                if cache_file:
                    cache_file.write(json.dumps({"item_id": item.id, "output": asdict(out)}) + "\n")
                    cache_file.flush()
            outputs.append(out)
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (total - i - 1) / rate if rate > 0 else float("nan")
            tag = " (cached)" if item.id in cached else ""
            print(
                f"[{detector.tool_id}] {i + 1}/{total} done{tag} | elapsed={elapsed:.0f}s | eta={eta:.0f}s",
                flush=True,
            )
    finally:
        if cache_file:
            cache_file.close()
    return outputs


class Detector(ABC):
    tool_id: str
    description: str
    expected_strength: str
    primary_cost: str

    @abstractmethod
    def predict(self, item: NewsItem) -> DetectorOutput: ...

    def predict_batch(self, items: List[NewsItem], cache_path: Optional[str] = None) -> List[DetectorOutput]:
        return predict_batch_with_cache(self, items, cache_path)

    def catalogue_entry(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "expected_strength": self.expected_strength,
            "primary_cost": self.primary_cost,
        }
