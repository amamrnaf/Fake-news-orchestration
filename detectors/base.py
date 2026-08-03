"""Common interface for every tool in the registry E = {T1, T2, T3} (Sec 3.4, Table 1)."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from utils.types import DetectorOutput, NewsItem


class Detector(ABC):
    tool_id: str
    description: str
    expected_strength: str
    primary_cost: str

    @abstractmethod
    def predict(self, item: NewsItem) -> DetectorOutput: ...

    def predict_batch(self, items: List[NewsItem]) -> List[DetectorOutput]:
        total = len(items)
        start = time.perf_counter()
        outputs = []
        for i, item in enumerate(items):
            outputs.append(self.predict(item))
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (total - i - 1) / rate if rate > 0 else float("nan")
            print(
                f"[{self.tool_id}] {i + 1}/{total} done | elapsed={elapsed:.0f}s | eta={eta:.0f}s",
                flush=True,
            )
        return outputs

    def catalogue_entry(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "expected_strength": self.expected_strength,
            "primary_cost": self.primary_cost,
        }
