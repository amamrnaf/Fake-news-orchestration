"""Semantic and affective encoders (Sec 3.2).

e_sem(z): sentence embedding from a general-purpose sentence encoder.
e_aff(z): affective representation — probabilities over an emotion/sentiment
          taxonomy from a dedicated affect classifier, used as the affective
          vector rather than a task-agnostic embedding, so it captures
          transferable affective signal (fear appeals, outrage, distrust,
          exaggerated certainty per Sec 8) rather than generic semantics.
"""
from __future__ import annotations

from typing import List

import numpy as np


class SemanticEncoder:
    """Wraps a sentence-transformers model to produce e_sem(x)."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False
        )


class AffectiveEncoder:
    """Wraps a multi-class emotion classifier to produce e_aff(x) as a probability vector.

    Default checkpoint outputs Ekman-style emotions (anger, disgust, fear, joy,
    neutral, sadness, surprise); any HF sequence-classification checkpoint with
    a softmax head can be substituted via `model_name`.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: str | None = None,
        max_length: int = 256,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_n @ b_n.T
