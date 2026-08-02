"""Common interface for every tool in the registry E = {T1, T2, T3} (Sec 3.4, Table 1)."""
from __future__ import annotations

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
        return [self.predict(item) for item in items]

    def catalogue_entry(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "expected_strength": self.expected_strength,
            "primary_cost": self.primary_cost,
        }
