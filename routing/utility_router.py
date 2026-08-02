"""Closed-form utility router realising Eq. in Sec 3.1:

    j_i* = argmax_j U(E_j | x_i, d_hat_i, a_i, C),   U(E_j) = Q(E_j) - lambda * Cost(E_j)

The primary experiments fix lambda=0 (pure predictive selection, delegated to
the LLM selector in `routing.llm_router`). This module implements the
*secondary efficiency experiment* that sweeps lambda to trace the
accuracy-cost trade-off (Sec 3.1, Sec 6 efficiency analysis) using the
validation catalogue directly, without an LLM call, so a lambda sweep is cheap
to run.
"""
from __future__ import annotations

from typing import Dict, List

from utils.types import DomainPrediction, RouterDecision, TOOL_IDS


def normalise_costs(mean_latency_by_tool: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalises per-tool latency to [0, 1] so lambda is comparable
    across tools with very different absolute costs (e.g. one encoder forward
    pass for T1 vs. generation + retrieval for T3)."""
    values = list(mean_latency_by_tool.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in mean_latency_by_tool}
    return {k: (v - lo) / (hi - lo) for k, v in mean_latency_by_tool.items()}


def utility_select(
    domain_pred: DomainPrediction,
    catalogue: dict,
    lam: float,
    item_id: str,
    router_name: str = "utility",
) -> RouterDecision:
    domain_entry = catalogue.get(domain_pred.predicted_domain) or catalogue.get("__all__", {})
    quality = {t: domain_entry.get(t, {}).get("macro_f1", 0.0) for t in TOOL_IDS if t in domain_entry}
    if not quality:
        quality = {t: 0.0 for t in TOOL_IDS}

    latencies = {t: domain_entry.get(t, {}).get("mean_latency_s", 0.0) for t in quality}
    costs = normalise_costs(latencies) if latencies else {t: 0.0 for t in quality}

    utilities = {t: quality[t] - lam * costs.get(t, 0.0) for t in quality}
    best_tool = max(utilities, key=utilities.get)

    return RouterDecision(
        item_id=item_id,
        selected_tool=best_tool,
        selection_confidence=1.0,
        reason_code="LOW_COST" if lam > 0 else "DOMAIN_FIT",
        short_justification=f"argmax utility (lambda={lam}): {utilities}",
        router_name=router_name,
        llm_calls=0,
        latency_s=0.0,
    )


def lambda_sweep(
    domain_preds: List[DomainPrediction],
    item_ids: List[str],
    catalogue: dict,
    lambdas: List[float],
) -> Dict[float, List[RouterDecision]]:
    return {
        lam: [utility_select(dp, catalogue, lam, item_id) for dp, item_id in zip(domain_preds, item_ids)]
        for lam in lambdas
    }
