"""T2: Retrieved-example LLM detector (Table 1) and the retrieval-mode baselines
of Table 4 (LLM zero-shot / + random examples / + semantic retrieval /
+ affective retrieval), which are all the same detector with different example
selection strategies.

'A 4-bit 7-8B instruction model prompted with top-K labelled examples.
Flexible cross-domain reasoning and adaptation without full fine-tuning.
Primary cost: generative inference and prompt tokens.'
"""
from __future__ import annotations

import random
from typing import List, Literal, Optional, Sequence

from detectors.base import predict_batch_with_cache
from detectors.llm_backend import LLMBackend
from retrieval.index import AffectiveSemanticIndex, RetrievedExample
from utils.json_schema import extract_json_block
from utils.text import truncate_chars
from utils.types import DetectorOutput, NewsItem

import json

RetrievalMode = Literal["zero_shot", "random", "semantic", "affective", "affective_semantic"]

_SYSTEM_PROMPT = (
    "You are a fake-news veracity classifier. Decide whether the target news "
    "item is FAKE or LEGITIMATE using only the target text and, if provided, "
    "the labelled examples. Return valid JSON only, matching this schema: "
    '{"label": "FAKE|LEGITIMATE", "confidence": 0.0, "explanation": "..."}'
)


def _format_examples(examples: Sequence[RetrievedExample]) -> str:
    if not examples:
        return "(no labelled examples provided)"
    lines = []
    for i, ex in enumerate(examples):
        gold = "FAKE" if ex.item.label == 1 else "LEGITIMATE"
        lines.append(f"Example {i+1} [{gold}]: {ex.item.text[:500]}")
    return "\n".join(lines)


def _build_prompt(item: NewsItem, examples: Sequence[RetrievedExample]) -> str:
    return (
        f"Labelled examples:\n{_format_examples(examples)}\n\n"
        f"Target item:\n{truncate_chars(item.text)}\n\n"
        "Respond with the JSON object only."
    )


class RetrievedExampleLLMDetector:
    """Not a `Detector` subclass because it needs a per-call retrieval-index
    argument at construction; it still implements the same `predict` contract
    (tool_id, description, ...) so it plugs into the same registry/catalogue
    code paths."""

    tool_id = "T2"
    description = "A 4-bit 7-8B instruction model prompted with top-K labelled examples."
    expected_strength = "Flexible cross-domain reasoning and adaptation without full fine-tuning."
    primary_cost = "Generative inference and prompt tokens."

    def __init__(
        self,
        llm: LLMBackend,
        index: Optional[AffectiveSemanticIndex] = None,
        semantic_encoder=None,
        affective_encoder=None,
        mode: RetrievalMode = "affective_semantic",
        k: int = 4,
        alpha: float = 0.5,
        max_new_tokens: int = 200,
        seed: int = 42,
    ):
        self.llm = llm
        self.index = index
        self.semantic_encoder = semantic_encoder
        self.affective_encoder = affective_encoder
        self.mode = mode
        self.k = k
        self.alpha = {"semantic": 1.0, "affective": 0.0, "affective_semantic": alpha}.get(mode, alpha)
        self.max_new_tokens = max_new_tokens
        self._rng = random.Random(seed)

    def _retrieve(self, item: NewsItem) -> List[RetrievedExample]:
        if self.mode == "zero_shot" or self.k == 0 or self.index is None:
            return []
        if self.mode == "random":
            pool = [it for it in self.index.items if it.id != item.id]
            n = min(self.k, len(pool))
            chosen = self._rng.sample(pool, n)
            return [RetrievedExample(item=it, score=0.0, sem_score=0.0, aff_score=0.0) for it in chosen]

        q_sem = self.semantic_encoder.encode([item.text])[0]
        q_aff = self.affective_encoder.encode([item.text])[0]
        return self.index.search(q_sem, q_aff, k=self.k, alpha=self.alpha, exclude_ids=[item.id])

    def catalogue_entry(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "expected_strength": self.expected_strength,
            "primary_cost": self.primary_cost,
        }

    def predict(self, item: NewsItem) -> DetectorOutput:
        examples = self._retrieve(item)
        prompt = _build_prompt(item, examples)
        meta = {"mode": self.mode, "k": self.k, "alpha": self.alpha, "n_examples": len(examples)}

        try:
            response = self.llm.generate(prompt, system=_SYSTEM_PROMPT, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            # A single backend error (e.g. context-length overflow, connection
            # drop) must not crash the whole predict_batch loop over the run.
            return DetectorOutput(
                tool_id=self.tool_id,
                label=0,
                prob_fake=0.5,
                explanation=None,
                failed=True,
                meta={**meta, "error": str(exc)},
            )

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
            explanation=explanation,
            latency_s=response.latency_s,
            generated_tokens=response.generated_tokens,
            peak_vram_mb=response.peak_vram_mb,
            failed=failed,
            meta=meta,
        )

    def predict_batch(self, items: List[NewsItem], cache_path: Optional[str] = None) -> List[DetectorOutput]:
        return predict_batch_with_cache(self, items, cache_path)
