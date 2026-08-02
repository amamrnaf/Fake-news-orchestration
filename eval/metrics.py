"""Veracity and domain classification metrics (Sec 4.5).

Veracity: macro-F1 (principal metric), accuracy, weighted F1, per-class
precision/recall, AUROC, Brier score, expected calibration error (ECE).
Domain: macro-F1, balanced accuracy, top-2 accuracy, confusion matrix.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def per_class_precision_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        str(label): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }


def auroc(y_true: Sequence[int], y_prob_fake: Sequence[float]) -> Optional[float]:
    y_true = np.asarray(y_true)
    if len(set(y_true.tolist())) < 2:
        return None  # AUROC undefined with a single class present
    return float(roc_auc_score(y_true, y_prob_fake))


def brier_score(y_true: Sequence[int], y_prob_fake: Sequence[float]) -> float:
    y_true, y_prob = np.asarray(y_true, dtype=float), np.asarray(y_prob_fake, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: Sequence[int], y_pred: Sequence[int], y_confidence: Sequence[float], n_bins: int = 10
) -> float:
    """Standard ECE over the confidence assigned to the *predicted* class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(y_confidence, dtype=float)
    correct = (y_true == y_pred).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(in_bin):
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def top2_accuracy(y_true_domains: Sequence[str], domain_distributions: Sequence[Dict[str, float]]) -> float:
    correct = 0
    for gold, dist in zip(y_true_domains, domain_distributions):
        top2 = sorted(dist, key=dist.get, reverse=True)[:2]
        correct += int(gold in top2)
    return correct / len(y_true_domains) if y_true_domains else 0.0


def domain_confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def veracity_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob_fake: Sequence[float],
    y_confidence: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """One-call summary matching the columns of Table 6 / Table 9 / Table 10."""
    conf = y_confidence if y_confidence is not None else [max(p, 1 - p) for p in y_prob_fake]
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "per_class": per_class_precision_recall(y_true, y_pred),
        "auroc": auroc(y_true, y_prob_fake),
        "brier": brier_score(y_true, y_prob_fake),
        "ece": expected_calibration_error(y_true, y_pred, conf),
    }


def domain_report(
    y_true_domains: Sequence[str],
    y_pred_domains: Sequence[str],
    domain_distributions: Sequence[Dict[str, float]],
    confidences: Sequence[float],
    labels: Sequence[str],
) -> Dict[str, object]:
    """Summary matching Table 5 columns."""
    correct = [int(t == p) for t, p in zip(y_true_domains, y_pred_domains)]
    return {
        "macro_f1": macro_f1(y_true_domains, y_pred_domains),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_domains, y_pred_domains)),
        "top2_accuracy": top2_accuracy(y_true_domains, domain_distributions),
        "ece": expected_calibration_error(
            [1] * len(correct), correct, confidences
        ),  # binary "was the domain prediction correct" calibration
        "confusion_matrix": domain_confusion_matrix(y_true_domains, y_pred_domains, labels).tolist(),
    }
