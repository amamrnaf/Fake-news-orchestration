"""Paired bootstrap significance testing with Holm correction (Sec 4.5):
'Paired bootstrap resampling with [10,000] samples is used for significance
testing, with Holm correction for multiple comparisons.'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    observed_diff: float
    p_value: float
    ci_low: float
    ci_high: float


def paired_bootstrap_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    metric_fn: Callable[[Sequence[int], Sequence[int]], float],
    n_samples: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Tests whether metric_fn(y_true, y_pred_a) differs from metric_fn(y_true, y_pred_b)
    on the same instances (paired), e.g. macro-F1 of the proposed router vs. the
    strongest individual detector (RQ1)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    observed = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    diffs = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])

    # Two-sided p-value: proportion of resampled diffs at least as extreme as
    # the null (zero), following the observed direction.
    if observed >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    alpha = 1 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(observed_diff=float(observed), p_value=p_value, ci_low=float(lo), ci_high=float(hi))


def holm_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Holm-Bonferroni step-down correction for multiple pairwise comparisons.
    Returns {comparison_name: (adjusted_p, reject_null)}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, Tuple[float, bool]] = {}
    max_adj_so_far = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min((m - rank) * p, 1.0)
        max_adj_so_far = max(max_adj_so_far, adj)  # monotonicity: adjusted p-values are non-decreasing
        adjusted[name] = (max_adj_so_far, max_adj_so_far < alpha)
    return adjusted
