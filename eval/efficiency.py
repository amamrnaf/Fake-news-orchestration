"""Efficiency accounting (Sec 4.5, Table 11): tools invoked per input, LLM
calls, generated tokens, latency, and peak GPU memory, plus cost relative to
the most expensive method (all-tool advisor, Table 11 = 1.00)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from utils.types import DetectorOutput


@dataclass
class EfficiencyRecord:
    item_id: str
    method: str
    tools_invoked: List[str] = field(default_factory=list)
    llm_calls: int = 0
    generated_tokens: int = 0
    latency_s: float = 0.0
    peak_vram_mb: float = 0.0

    @property
    def tools_per_input(self) -> int:
        return len(self.tools_invoked)


def record_from_outputs(item_id: str, method: str, outputs: List[DetectorOutput], router_llm_calls: int = 0) -> EfficiencyRecord:
    llm_tool_ids = {"T2", "T3"}
    return EfficiencyRecord(
        item_id=item_id,
        method=method,
        tools_invoked=[o.tool_id for o in outputs],
        llm_calls=router_llm_calls + sum(1 for o in outputs if o.tool_id in llm_tool_ids),
        generated_tokens=sum(o.generated_tokens for o in outputs),
        latency_s=sum(o.latency_s for o in outputs),
        peak_vram_mb=max([o.peak_vram_mb for o in outputs], default=0.0),
    )


def aggregate_efficiency(records: List[EfficiencyRecord]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    return {
        "seconds_per_item": sum(r.latency_s for r in records) / n,
        "generated_tokens": sum(r.generated_tokens for r in records) / n,
        "peak_vram_mb": max((r.peak_vram_mb for r in records), default=0.0),
        "tools_per_input": sum(r.tools_per_input for r in records) / n,
        "llm_calls": sum(r.llm_calls for r in records) / n,
    }


def relative_cost(method_cost: float, reference_cost: float) -> float:
    """Table 11's 'Relative cost' column, normalised so the all-tool advisor = 1.00."""
    if reference_cost == 0:
        return 0.0
    return method_cost / reference_cost
