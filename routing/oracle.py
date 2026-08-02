"""Oracle router and selection regret (Sec 3.7).

'An oracle router selects a detector that predicts the gold label correctly
whenever at least one detector is correct. The oracle is not deployable; it
measures the maximum achievable accuracy of the existing tool bank.'

R_i = max_j 1[y_hat_ij = y_i] - 1[y_hat_i,selected = y_i]
"""
from __future__ import annotations

from typing import Dict, List, Set

from utils.types import DetectorOutput, NewsItem, RouterDecision, TOOL_IDS


def correct_tools_per_item(items: List[NewsItem], outputs_by_tool: Dict[str, List[DetectorOutput]]) -> List[Set[str]]:
    """For each item, the set of tool_ids whose prediction matched the gold label."""
    n = len(items)
    result: List[Set[str]] = [set() for _ in range(n)]
    for tool_id, outputs in outputs_by_tool.items():
        for i, (item, out) in enumerate(zip(items, outputs)):
            if out.label == item.label:
                result[i].add(tool_id)
    return result


def oracle_select(items: List[NewsItem], outputs_by_tool: Dict[str, List[DetectorOutput]]) -> List[RouterDecision]:
    """Picks any correct tool when one exists; otherwise defaults to T1
    (arbitrary but deterministic, since no tool is correct and the choice
    cannot change the outcome)."""
    correct_sets = correct_tools_per_item(items, outputs_by_tool)
    decisions = []
    for item, correct in zip(items, correct_sets):
        chosen = next(iter(sorted(correct))) if correct else "T1"
        decisions.append(
            RouterDecision(
                item_id=item.id,
                selected_tool=chosen,
                selection_confidence=1.0,
                reason_code="OTHER",
                short_justification="Oracle: any tool correct on this instance" if correct else "Oracle: no tool correct",
                router_name="oracle",
                llm_calls=0,
                latency_s=0.0,
            )
        )
    return decisions


def outputs_from_decisions(
    decisions: List[RouterDecision], outputs_by_tool: Dict[str, List[DetectorOutput]]
) -> List[DetectorOutput]:
    """Materialises the DetectorOutput actually returned by whichever tool
    each decision selected, for a given item order."""
    return [outputs_by_tool[d.selected_tool][i] for i, d in enumerate(decisions)]
