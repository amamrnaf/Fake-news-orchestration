"""Shared data structures used across the pipeline.

Kept dependency-free (stdlib only) so every module can import them
without pulling in torch/transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DOMAINS = [
    "technology",
    "education",
    "business",
    "sports",
    "politics",
    "entertainment",
    "celebrity",
]

TOOL_IDS = ["T1", "T2", "T3"]


@dataclass
class NewsItem:
    """One news/claim instance shared by all datasets (AMTCele, PHEME, COCO)."""

    id: str
    text: str
    label: int  # 1 = fake/rumour/conspiracy, 0 = legitimate/non-rumour/related
    domain: Optional[str] = None  # AMTCele topical domain, or None for PHEME/COCO
    dataset: str = "amtcele"  # amtcele | pheme | coco
    split: Optional[str] = None  # train | val | test
    group: Optional[str] = None  # event id / thread id / topic group for leakage control
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    """Standard output contract for every tool in the registry (T1/T2/T3)."""

    tool_id: str
    label: int
    prob_fake: float  # p(y=1 | x)
    evidence: Optional[str] = None
    explanation: Optional[str] = None
    latency_s: float = 0.0
    generated_tokens: int = 0
    peak_vram_mb: float = 0.0
    failed: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainPrediction:
    distribution: Dict[str, float]  # domain -> probability, sums to 1
    predicted_domain: str
    entropy: float
    confidence: float  # max probability


@dataclass
class RouterDecision:
    item_id: str
    selected_tool: str  # "T1" | "T2" | "T3"
    selection_confidence: float
    reason_code: str  # DOMAIN_FIT | EVIDENCE_REQUIRED | OOD_REASONING | LOW_COST | OTHER
    short_justification: str
    router_name: str = "proposed"
    llm_calls: int = 1
    latency_s: float = 0.0
    raw_response: Optional[str] = None
