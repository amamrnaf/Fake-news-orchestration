"""LLM tool selector (Sec 3.6): the proposed router. Two modes:

  - "domain_catalogue": sees only predicted domain (+ distribution/entropy)
    and the validation catalogue.
  - "instance_aware": additionally sees claim features (Sec 3.6: named-entity
    density, numerical expressions, temporal cues, retrieval similarity) via
    `routing.features.compute_instance_features`.

Retries on schema-invalid output before falling back to the catalogue's best
validation tool for the predicted domain, so a single malformed generation
can't silently corrupt downstream metrics.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Literal, Optional

from detectors.llm_backend import LLMBackend
from retrieval.index import RetrievedExample
from routing.features import InstanceFeatures, compute_instance_features
from routing.selector_prompt import SELECTOR_SYSTEM_PROMPT, build_selector_user_prompt
from utils.json_schema import RouterSchemaError, parse_router_output
from utils.types import DomainPrediction, NewsItem, RouterDecision

RoutingMode = Literal["domain_catalogue", "instance_aware"]


class LLMRouter:
    def __init__(
        self,
        llm: LLMBackend,
        tool_descriptions: List[dict],
        validation_catalogue: dict,
        mode: RoutingMode = "instance_aware",
        max_retries: int = 2,
        max_new_tokens: int = 150,
    ):
        self.llm = llm
        self.tool_descriptions = tool_descriptions
        self.validation_catalogue = validation_catalogue
        self.mode = mode
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens

    def _fallback_tool(self, domain: Optional[str]) -> str:
        entry = self.validation_catalogue.get(domain) or self.validation_catalogue.get("__all__", {})
        return entry.get("best_validation_tool", "T1")

    def select(
        self,
        item: NewsItem,
        domain_pred: DomainPrediction,
        affective_summary: Dict[str, float],
        retrieved_examples: List[RetrievedExample],
        router_name: str = "proposed",
    ) -> RouterDecision:
        top_k_meta = [
            {"domain": r.item.domain, "label": "FAKE" if r.item.label == 1 else "LEGITIMATE", "score": r.score}
            for r in retrieved_examples
        ]
        domain_catalogue_slice = self.validation_catalogue.get(domain_pred.predicted_domain, {})

        instance_features = None
        if self.mode == "instance_aware":
            feats: InstanceFeatures = compute_instance_features(item.text, domain_pred, retrieved_examples)
            instance_features = asdict(feats)

        user_prompt = build_selector_user_prompt(
            target_text=item.text,
            predicted_domain_distribution=domain_pred.distribution,
            domain_entropy=domain_pred.entropy,
            affective_summary=affective_summary,
            top_k_example_metadata=top_k_meta,
            available_tools=self.tool_descriptions,
            validation_performance_catalogue=domain_catalogue_slice,
            instance_features=instance_features,
        )

        llm_calls = 0
        start = time.perf_counter()
        last_error: Optional[str] = None
        raw_text = ""
        for _ in range(self.max_retries + 1):
            response = self.llm.generate(user_prompt, system=SELECTOR_SYSTEM_PROMPT, max_new_tokens=self.max_new_tokens)
            raw_text = response.text
            llm_calls += 1
            try:
                selection = parse_router_output(raw_text)
                latency = time.perf_counter() - start
                return RouterDecision(
                    item_id=item.id,
                    selected_tool=selection.selected_tool,
                    selection_confidence=selection.selection_confidence,
                    reason_code=selection.reason_code,
                    short_justification=selection.short_justification,
                    router_name=router_name,
                    llm_calls=llm_calls,
                    latency_s=latency,
                    raw_response=raw_text,
                )
            except RouterSchemaError as exc:
                last_error = str(exc)
                continue

        # All retries exhausted: fall back to the validation catalogue's best
        # tool for the predicted domain rather than crashing the pipeline.
        latency = time.perf_counter() - start
        return RouterDecision(
            item_id=item.id,
            selected_tool=self._fallback_tool(domain_pred.predicted_domain),
            selection_confidence=0.0,
            reason_code="OTHER",
            short_justification=f"Fallback after schema failure: {last_error}",
            router_name=router_name,
            llm_calls=llm_calls,
            latency_s=latency,
            raw_response=raw_text,
        )
