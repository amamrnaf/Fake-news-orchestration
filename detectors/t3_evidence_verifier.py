"""T3: Evidence-grounded verifier (Table 1).

'Retriever plus LLM decision using relevant factual passages or claim
matches. Entity-specific and factual claims that require explicit evidence.
Primary cost: retrieval, reranking, and generation.'

Retrieval is BM25 over a user-supplied evidence corpus (fact-check archive,
Wikipedia snapshot, or verified-claim database — any list of passages).
Failure to retrieve a passage above `min_score` is treated as evidence-retrieval
failure (Sec 7.3) rather than forcing a guess: the detector abstains toward the
majority/legitimate class with low confidence and `failed=True` so error
analysis can separate "no evidence found" from "wrong decision".
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from rank_bm25 import BM25Okapi

from detectors.llm_backend import LLMBackend
from utils.json_schema import extract_json_block
from utils.types import DetectorOutput, NewsItem

_SYSTEM_PROMPT = (
    "You are an evidence-grounded fact verifier. Decide whether the target "
    "claim is FAKE or LEGITIMATE using only the retrieved passages below. If "
    "the passages neither support nor contradict the claim, say so explicitly. "
    'Return valid JSON only: {"label": "FAKE|LEGITIMATE", "confidence": 0.0, '
    '"supported": true|false, "explanation": "..."}'
)


@dataclass
class EvidencePassage:
    doc_id: str
    text: str


class EvidenceCorpus:
    """BM25 index over a flat evidence corpus."""

    def __init__(self, passages: Sequence[EvidencePassage]):
        self.passages = list(passages)
        tokenized = [p.text.lower().split() for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0) -> List[EvidencePassage]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.passages[i] for i in ranked if scores[i] >= min_score]


class EvidenceGroundedVerifier:
    tool_id = "T3"
    description = "Retriever plus LLM decision using relevant factual passages or claim matches."
    expected_strength = "Entity-specific and factual claims that require explicit evidence."
    primary_cost = "Retrieval, reranking, and generation."

    def __init__(
        self,
        llm: LLMBackend,
        corpus: EvidenceCorpus,
        k: int = 5,
        min_score: float = 0.0,
        max_new_tokens: int = 220,
    ):
        self.llm = llm
        self.corpus = corpus
        self.k = k
        self.min_score = min_score
        self.max_new_tokens = max_new_tokens

    def catalogue_entry(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "expected_strength": self.expected_strength,
            "primary_cost": self.primary_cost,
        }

    def predict(self, item: NewsItem) -> DetectorOutput:
        start = time.perf_counter()
        passages = self.corpus.retrieve(item.text, k=self.k, min_score=self.min_score)
        retrieval_latency = time.perf_counter() - start

        if not passages:
            # Sec 7.3: evidence retrieval failure -> abstain, do not fabricate support.
            return DetectorOutput(
                tool_id=self.tool_id,
                label=0,
                prob_fake=0.5,
                evidence=None,
                explanation="No relevant evidence retrieved; abstained toward legitimate with low confidence.",
                latency_s=retrieval_latency,
                generated_tokens=0,
                failed=True,
                meta={"n_passages": 0},
            )

        evidence_text = "\n".join(f"[{p.doc_id}] {p.text[:500]}" for p in passages)
        prompt = f"Retrieved passages:\n{evidence_text}\n\nClaim:\n{item.text}\n\nRespond with the JSON object only."
        response = self.llm.generate(prompt, system=_SYSTEM_PROMPT, max_new_tokens=self.max_new_tokens)

        try:
            payload = json.loads(extract_json_block(response.text))
            label = 1 if str(payload.get("label", "")).strip().upper() == "FAKE" else 0
            confidence = float(payload.get("confidence", 0.5))
            explanation = payload.get("explanation")
            failed = False
        except Exception:
            label, confidence, explanation, failed = 0, 0.5, None, True

        prob_fake = confidence if label == 1 else 1.0 - confidence
        return DetectorOutput(
            tool_id=self.tool_id,
            label=label,
            prob_fake=prob_fake,
            evidence=evidence_text,
            explanation=explanation,
            latency_s=retrieval_latency + response.latency_s,
            generated_tokens=response.generated_tokens,
            peak_vram_mb=response.peak_vram_mb,
            failed=failed,
            meta={"n_passages": len(passages)},
        )

    def predict_batch(self, items: List[NewsItem]) -> List[DetectorOutput]:
        return [self.predict(it) for it in items]
