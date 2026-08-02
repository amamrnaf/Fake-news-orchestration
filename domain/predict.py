"""Inference for the domain classifier: returns the full distribution p(d|x),
its entropy, and max confidence (Sec 3.3), which the selector consumes to
reflect ambiguity (e.g. entertainment vs. celebrity)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from domain.model import DomainClassifierModel, build_domain_input_text
from utils.types import DOMAINS, DomainPrediction


def _entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def predict_domain_batch(
    clf: DomainClassifierModel,
    texts: Sequence[str],
    neighbours: Optional[List] = None,
    batch_size: int = 32,
) -> List[DomainPrediction]:
    import torch

    clf.model.eval()
    input_texts = [
        build_domain_input_text(t, neighbours[i] if neighbours else None) for i, t in enumerate(texts)
    ]
    predictions: List[DomainPrediction] = []
    with torch.no_grad():
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i : i + batch_size]
            enc = clf.tokenize(batch)
            logits = clf.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                dist = {DOMAINS[j]: float(row[j]) for j in range(len(DOMAINS))}
                top = int(row.argmax())
                predictions.append(
                    DomainPrediction(
                        distribution=dist,
                        predicted_domain=DOMAINS[top],
                        entropy=_entropy(row),
                        confidence=float(row[top]),
                    )
                )
    return predictions


def predict_domain_one(clf: DomainClassifierModel, text: str, neighbours: Optional[List] = None) -> DomainPrediction:
    return predict_domain_batch(clf, [text], [neighbours] if neighbours is not None else None)[0]
