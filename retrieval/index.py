"""Affective-semantic retrieval index (Sec 3.2, Eq. S(x,z)).

S(x,z) = alpha * cos(e_sem(x), e_sem(z)) + (1-alpha) * cos(e_aff(x), e_aff(z))

Implemented as a single inner-product FAISS index over the concatenated,
per-branch L2-normalised and alpha-scaled vector
    v(*) = [sqrt(alpha) * sem_norm(*), sqrt(1-alpha) * aff_norm(*)]
so that <v(x), v(z)> == S(x,z) exactly, letting FAISS do the top-K search.
Rebuilding the index when alpha changes is cheap because the source pool
(validation/train partitions used for retrieval) is small (Sec 9, "AMTCele is
small").

alpha=1 -> semantic-only retrieval (RQ2 ablation).
alpha=0 -> affective-only retrieval (RQ2 ablation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from utils.types import NewsItem

NEAR_DUPLICATE_THRESHOLD = 0.95
_NEAR_DUPLICATE_SLACK = 16


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


@dataclass
class RetrievedExample:
    item: NewsItem
    score: float
    sem_score: float
    aff_score: float


class AffectiveSemanticIndex:
    """Index over a fixed source pool (e.g. train + val partitions excluding the
    held-out domain, per Sec 4.2 leave-one-domain-out leakage control)."""

    def __init__(self, items: Sequence[NewsItem], sem_embeddings: np.ndarray, aff_embeddings: np.ndarray):
        assert len(items) == len(sem_embeddings) == len(aff_embeddings)
        self.items = list(items)
        self.sem = _l2_normalise(np.asarray(sem_embeddings, dtype="float32"))
        self.aff = _l2_normalise(np.asarray(aff_embeddings, dtype="float32"))
        self._alpha: float | None = None
        self._faiss_index = None

    def _build(self, alpha: float) -> None:
        import faiss

        w_sem = np.sqrt(alpha)
        w_aff = np.sqrt(1.0 - alpha)
        vecs = np.concatenate([w_sem * self.sem, w_aff * self.aff], axis=1).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._faiss_index = index
        self._alpha = alpha
        self._vecs = vecs

    def search(
        self,
        query_sem: np.ndarray,
        query_aff: np.ndarray,
        k: int,
        alpha: float,
        exclude_ids: Sequence[str] = (),
        near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    ) -> List[RetrievedExample]:
        if self._alpha != alpha:
            self._build(alpha)

        q_sem = _l2_normalise(np.asarray(query_sem, dtype="float32").reshape(1, -1))
        q_aff = _l2_normalise(np.asarray(query_aff, dtype="float32").reshape(1, -1))
        q = np.concatenate([np.sqrt(alpha) * q_sem, np.sqrt(1.0 - alpha) * q_aff], axis=1).astype("float32")

        fetch_k = min(len(self.items), k + len(exclude_ids) + _NEAR_DUPLICATE_SLACK)
        scores, idxs = self._faiss_index.search(q, fetch_k)
        scores, idxs = scores[0], idxs[0]

        exclude = set(exclude_ids)
        results: List[RetrievedExample] = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            item = self.items[idx]
            if item.id in exclude:
                continue
            sem_score = float(self.sem[idx] @ q_sem[0])
            # Never return the query itself or a near-duplicate of it: when the
            # query is in the pool (validation items, which share the pool with
            # training), it otherwise retrieves itself with its gold label.
            # Measured: 147/147 validation items did, inflating the
            # retrieval-augmented detector's validation macro-F1 to 0.939 vs
            # 0.752 on test. Same threshold and encoder as the split's
            # near-duplicate grouping, so test queries are unaffected.
            if sem_score >= near_duplicate_threshold:
                continue
            aff_score = float(self.aff[idx] @ q_aff[0])
            results.append(RetrievedExample(item=item, score=float(score), sem_score=sem_score, aff_score=aff_score))
            if len(results) >= k:
                break
        return results

    def search_batch(
        self,
        query_sem: np.ndarray,
        query_aff: np.ndarray,
        k: int,
        alpha: float,
        exclude_ids_per_query: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[List[RetrievedExample]]:
        """`exclude_ids_per_query[i]` is passed to `search` as `exclude_ids` for
        query i. Needed whenever the queries are themselves in the pool (for
        example routing validation items over a train+validation index)."""
        return [
            self.search(
                query_sem[i], query_aff[i], k=k, alpha=alpha,
                exclude_ids=exclude_ids_per_query[i] if exclude_ids_per_query is not None else (),
            )
            for i in range(query_sem.shape[0])
        ]


def build_index_for_pool(
    pool_items: Sequence[NewsItem], semantic_encoder, affective_encoder
) -> AffectiveSemanticIndex:
    """Convenience builder: embeds a source pool and wraps it in an index.
    `pool_items` must already exclude the target/held-out partition (Sec 3.2:
    'Test labels are never included in the index.')."""
    texts = [it.text for it in pool_items]
    sem = semantic_encoder.encode(texts)
    aff = affective_encoder.encode(texts)
    return AffectiveSemanticIndex(pool_items, sem, aff)
