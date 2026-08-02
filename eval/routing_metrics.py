"""Routing-quality metrics (Sec 3.7, Sec 4.5, Table 8, Table 7 oracle gap).

selected-tool correctness, agreement with the best validation tool, oracle
gap, mean selection regret, tool-selection distribution, and the
oracle/best-tool-vs-selected-tool confusion matrix requested after Table 8.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from utils.types import TOOL_IDS


def selection_accuracy(selected_tools: Sequence[str], best_tools: Sequence[str]) -> float:
    """Fraction of instances where the router selected a tool that was
    correct on that instance (best_tools here should be "any correct tool",
    see `oracle.correct_tools_per_item`)."""
    return float(np.mean([s in b if isinstance(b, (set, list, tuple)) else s == b for s, b in zip(selected_tools, best_tools)]))


def agreement_with_best_validation_tool(selected_tools: Sequence[str], best_validation_tool: str) -> float:
    """Agreement with the single best tool on the *validation* catalogue,
    i.e. what a domain-rule router would always pick (Sec 4.3 Domain-rule
    router). `best_validation_tool` may instead be a per-item domain-indexed
    best tool passed in as a list."""
    if isinstance(best_validation_tool, str):
        targets = [best_validation_tool] * len(selected_tools)
    else:
        targets = best_validation_tool
    return float(np.mean([s == t for s, t in zip(selected_tools, targets)]))


def oracle_gap(router_macro_f1: float, oracle_macro_f1: float) -> float:
    return float(oracle_macro_f1 - router_macro_f1)


def mean_selection_regret(
    per_item_correctness: Sequence[Dict[str, bool]], selected_tools: Sequence[str]
) -> float:
    """Router regret (Eq. Sec 3.7): R_i = max_j 1[correct_j] - 1[correct_selected].
    `per_item_correctness[i]` maps tool_id -> whether that tool was correct on item i."""
    regrets = []
    for correctness, selected in zip(per_item_correctness, selected_tools):
        best = max(int(v) for v in correctness.values()) if correctness else 0
        chosen = int(correctness.get(selected, False))
        regrets.append(best - chosen)
    return float(np.mean(regrets)) if regrets else 0.0


def tool_selection_distribution(selected_tools: Sequence[str]) -> Dict[str, float]:
    counts = Counter(selected_tools)
    n = len(selected_tools)
    return {tool: counts.get(tool, 0) / n for tool in TOOL_IDS} if n else {t: 0.0 for t in TOOL_IDS}


def selection_confusion_matrix(
    row_labels: Sequence[str], selected_tools: Sequence[str], tools: Sequence[str] = TOOL_IDS
) -> np.ndarray:
    """Rows = oracle/best tool, columns = selected tool (per the paper's
    post-Table-8 instruction: 'ADD A SELECTION CONFUSION MATRIX: ROWS =
    ORACLE/BEST TOOL; COLUMNS = SELECTED TOOL.')."""
    idx = {t: i for i, t in enumerate(tools)}
    mat = np.zeros((len(tools), len(tools)), dtype=int)
    for row, col in zip(row_labels, selected_tools):
        if row in idx and col in idx:
            mat[idx[row], idx[col]] += 1
    return mat
