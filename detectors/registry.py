"""Detector registry E = {T1, T2, T3} (Sec 3.4, Table 1)."""
from __future__ import annotations

from typing import Dict, Union

from detectors.t1_supervised import SupervisedTransformerDetector
from detectors.t2_retrieval_llm import RetrievedExampleLLMDetector
from detectors.t3_evidence_verifier import EvidenceGroundedVerifier

AnyDetector = Union[SupervisedTransformerDetector, RetrievedExampleLLMDetector, EvidenceGroundedVerifier]


class DetectorRegistry:
    def __init__(self, t1: SupervisedTransformerDetector, t2: RetrievedExampleLLMDetector, t3: EvidenceGroundedVerifier):
        self.tools: Dict[str, AnyDetector] = {"T1": t1, "T2": t2, "T3": t3}

    def get(self, tool_id: str) -> AnyDetector:
        return self.tools[tool_id]

    def all_ids(self):
        return list(self.tools.keys())

    def catalogue_descriptions(self) -> list[dict]:
        return [tool.catalogue_entry() for tool in self.tools.values()]
