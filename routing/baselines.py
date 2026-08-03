"""Baselines used to isolate retrieval, routing, and ensemble effects (Table 4):
majority class, random router, domain-rule router, confidence-weighted
ensemble, all-tool LLM advisor. (LLM zero-shot / +random/+semantic/+affective
retrieval baselines live in `detectors.t2_retrieval_llm` as retrieval modes
of the same detector.)
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from typing import Dict, List, Sequence

from detectors.llm_backend import LLMBackend
from utils.json_schema import extract_json_block
from utils.text import truncate_chars
from utils.types import DetectorOutput, DomainPrediction, NewsItem, RouterDecision, TOOL_IDS


def majority_class_label(train_items: Sequence[NewsItem]) -> int:
    counts = Counter(it.label for it in train_items)
    return counts.most_common(1)[0][0]


def majority_class_predictions(train_items: Sequence[NewsItem], n_test: int) -> List[DetectorOutput]:
    label = majority_class_label(train_items)
    prob_fake = 1.0 if label == 1 else 0.0
    return [DetectorOutput(tool_id="MAJORITY", label=label, prob_fake=prob_fake) for _ in range(n_test)]


class RandomRouter:
    """Lower bound for detector selection (Table 4)."""

    def __init__(self, seed: int = 42, tools: Sequence[str] = TOOL_IDS):
        self._rng = random.Random(seed)
        self.tools = list(tools)

    def select(self, item: NewsItem) -> RouterDecision:
        chosen = self._rng.choice(self.tools)
        return RouterDecision(
            item_id=item.id,
            selected_tool=chosen,
            selection_confidence=1.0 / len(self.tools),
            reason_code="OTHER",
            short_justification="Uniform random selection.",
            router_name="random",
            llm_calls=0,
        )


class DomainRuleRouter:
    """Selects the best validation detector for the predicted domain (Table 4)."""

    def __init__(self, validation_catalogue: dict):
        self.catalogue = validation_catalogue

    def select(self, item: NewsItem, domain_pred: DomainPrediction) -> RouterDecision:
        entry = self.catalogue.get(domain_pred.predicted_domain) or self.catalogue.get("__all__", {})
        best_tool = entry.get("best_validation_tool", "T1")
        return RouterDecision(
            item_id=item.id,
            selected_tool=best_tool,
            selection_confidence=domain_pred.confidence,
            reason_code="DOMAIN_FIT",
            short_justification=f"Best validation tool for domain={domain_pred.predicted_domain!r}.",
            router_name="domain_rule",
            llm_calls=0,
        )


def confidence_weighted_ensemble(outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
    """Combines all detector probabilities, weighted by each tool's distance
    from 0.5 (a simple proxy for confidence) — Table 4's 'Confidence-weighted
    ensemble' baseline. Failed detector calls are excluded from the mix."""
    weighted_sum, weight_total = 0.0, 0.0
    latency, tokens, vram = 0.0, 0, 0.0
    for out in outputs.values():
        latency += out.latency_s
        tokens += out.generated_tokens
        vram = max(vram, out.peak_vram_mb)
        if out.failed:
            continue
        weight = abs(out.prob_fake - 0.5) * 2.0
        weighted_sum += weight * out.prob_fake
        weight_total += weight

    prob_fake = (weighted_sum / weight_total) if weight_total > 0 else 0.5
    label = int(prob_fake >= 0.5)
    return DetectorOutput(
        tool_id="ENSEMBLE",
        label=label,
        prob_fake=prob_fake,
        latency_s=latency,
        generated_tokens=tokens,
        peak_vram_mb=vram,
        meta={"n_tools": len(outputs)},
    )


_ADVISOR_SYSTEM_PROMPT = (
    "You are a synthesis advisor. Three detectors have each analysed the same "
    "news item and returned a label, confidence, and rationale. Combine their "
    "outputs into one final decision. Weigh disagreement and evidence quality. "
    'Return valid JSON only: {"label": "FAKE|LEGITIMATE", "confidence": 0.0, '
    '"explanation": "..."}'
)


def _format_tool_outputs(outputs: Dict[str, DetectorOutput]) -> str:
    lines = []
    for tool_id, out in outputs.items():
        label = "FAKE" if out.label == 1 else "LEGITIMATE"
        conf = max(out.prob_fake, 1 - out.prob_fake)
        rationale = out.explanation or out.evidence or "(no rationale)"
        lines.append(f"{tool_id}: label={label}, confidence={conf:.2f}, rationale={rationale[:300]}")
    return "\n".join(lines)


class AllToolLLMAdvisor:
    """Runs all tools and asks an LLM advisor to synthesise their outputs (Table 4)."""

    def __init__(self, llm: LLMBackend, max_new_tokens: int = 200):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def synthesise(self, item: NewsItem, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        prompt = (
            f"Target item:\n{truncate_chars(item.text)}\n\nDetector outputs:\n{_format_tool_outputs(outputs)}\n\n"
            "Respond with the JSON object only."
        )
        upstream_latency = sum(o.latency_s for o in outputs.values())
        upstream_tokens = sum(o.generated_tokens for o in outputs.values())
        try:
            response = self.llm.generate(prompt, system=_ADVISOR_SYSTEM_PROMPT, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return DetectorOutput(
                tool_id="ADVISOR",
                label=0,
                prob_fake=0.5,
                explanation=f"LLM backend error; abstained: {exc}",
                latency_s=upstream_latency,
                generated_tokens=upstream_tokens,
                failed=True,
                meta={"error": str(exc)},
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
            tool_id="ADVISOR",
            label=label,
            prob_fake=prob_fake,
            explanation=explanation,
            latency_s=upstream_latency + response.latency_s,
            generated_tokens=upstream_tokens + response.generated_tokens,
            peak_vram_mb=max([o.peak_vram_mb for o in outputs.values()] + [response.peak_vram_mb]),
            failed=failed,
        )
