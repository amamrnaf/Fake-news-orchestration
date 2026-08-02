"""Instance-level claim features used by instance-aware routing (Sec 3.6):
named-entity density, numerical expressions, temporal cues, domain entropy,
and retrieval similarity. Domain-catalogue routing uses none of these; it
only sees the predicted domain and the catalogue."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from retrieval.index import RetrievedExample
from utils.types import DomainPrediction

_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?%?\b")
_TEMPORAL_WORDS = {
    "today",
    "yesterday",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "this week",
    "last week",
    "next week",
    "this year",
    "last year",
}
_TEMPORAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _TEMPORAL_WORDS) + r")\b|\b\d{4}\b", re.IGNORECASE
)

_nlp = None  # lazily-loaded spaCy pipeline, shared across calls


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Install with: "
                "python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


@dataclass
class InstanceFeatures:
    named_entity_density: float  # entities per 100 tokens
    numerical_expression_count: int
    has_temporal_cue: bool
    domain_entropy: float
    retrieval_similarity_mean: float
    retrieval_similarity_max: float


def named_entity_density(text: str) -> float:
    doc = _get_spacy()(text)
    n_tokens = max(len(doc), 1)
    return 100.0 * len(doc.ents) / n_tokens


def numerical_expression_count(text: str) -> int:
    return len(_NUMBER_RE.findall(text))


def has_temporal_cue(text: str) -> bool:
    return bool(_TEMPORAL_RE.search(text))


def compute_instance_features(
    text: str,
    domain_pred: DomainPrediction,
    retrieved: Optional[List[RetrievedExample]] = None,
) -> InstanceFeatures:
    sims = [r.score for r in retrieved] if retrieved else []
    return InstanceFeatures(
        named_entity_density=named_entity_density(text),
        numerical_expression_count=numerical_expression_count(text),
        has_temporal_cue=has_temporal_cue(text),
        domain_entropy=domain_pred.entropy,
        retrieval_similarity_mean=float(sum(sims) / len(sims)) if sims else 0.0,
        retrieval_similarity_max=float(max(sims)) if sims else 0.0,
    )
