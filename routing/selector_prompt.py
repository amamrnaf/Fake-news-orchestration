"""Draft tool-selection prompt, reproduced verbatim from Appendix A of the
paper. [INSERT FINAL SELECTOR PROMPT IN APPENDIX A AFTER PILOT TESTING] —
this is the pilot version; replace `SELECTOR_SYSTEM_PROMPT` once the paper's
prompt-selection pilot is finalised, without changing the JSON schema
enforced by `utils.json_schema`.

Rule 3 of the original pilot ("Select the evidence verifier for
entity-specific, numerical, temporal, or externally checkable claims when
reliable retrieval is available") was removed after Table 8's confusion
matrix traced T3 over-selection to it: its surface-feature triggers (named
entities, numbers, dates) match most ordinary news text, so it fired far
more often than T3 was actually the better tool, independent of the
validation catalogue's measured performance. Table 7's LODO results showed
the same pattern -- sports/entertainment, the most "checkable-sounding"
domains, had the router's largest losses. Tool selection is now driven
solely by the supplied validation catalogue (Rule 1) rather than a
text-feature heuristic layered on top of it.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

SELECTOR_SYSTEM_PROMPT = """You are a routing agent. You do not decide whether the news item is true or false. Your task is to select the verification tool most likely to make a correct decision.
Input fields:
- target_text
- predicted_domain_distribution
- domain_entropy
- affective_summary
- top_k_example_metadata
- available_tools
- validation_performance_catalogue
Rules:
1. Use only the supplied tool descriptions and validation catalogue.
2. Prefer the lowest-cost tool when expected predictive quality is similar.
3. Do not output a veracity label.
4. Return valid JSON only.
Output schema:
{
  "selected_tool": "T1|T2|T3",
  "selection_confidence": 0.0,
  "reason_code": "DOMAIN_FIT|EVIDENCE_REQUIRED|OOD_REASONING|LOW_COST|OTHER",
  "short_justification": "..."
}"""

# The original pilot prompt, kept verbatim so the "original-prompt router"
# (PROTOCOL.md section 5, router variant 1) can be rerun on the clean catalogue.
ORIGINAL_SELECTOR_SYSTEM_PROMPT = SELECTOR_SYSTEM_PROMPT.replace(
    """Rules:
1. Use only the supplied tool descriptions and validation catalogue.
2. Prefer the lowest-cost tool when expected predictive quality is similar.
3. Do not output a veracity label.
4. Return valid JSON only.""",
    """Rules:
1. Use only the supplied tool descriptions and validation catalogue.
2. Prefer the lowest-cost tool when expected predictive quality is similar.
3. Select the evidence verifier for entity-specific, numerical, temporal, or externally checkable claims when reliable retrieval is available.
4. Do not output a veracity label.
5. Return valid JSON only.""",
)
assert ORIGINAL_SELECTOR_SYSTEM_PROMPT != SELECTOR_SYSTEM_PROMPT


def build_selector_user_prompt(
    target_text: str,
    predicted_domain_distribution: Dict[str, float],
    domain_entropy: float,
    affective_summary: Dict[str, float],
    top_k_example_metadata: List[dict],
    available_tools: List[dict],
    validation_performance_catalogue: dict,
    instance_features: Optional[dict] = None,
) -> str:
    """Serialises the Appendix A input fields as JSON for the user turn.

    `instance_features` (named-entity density, numerical expressions,
    temporal cues, retrieval similarity) is only populated in instance-aware
    routing mode (Sec 3.6); domain-catalogue routing passes `None`.
    """
    payload = {
        "target_text": target_text,
        "predicted_domain_distribution": predicted_domain_distribution,
        "domain_entropy": domain_entropy,
        "affective_summary": affective_summary,
        "top_k_example_metadata": top_k_example_metadata,
        "available_tools": available_tools,
        "validation_performance_catalogue": validation_performance_catalogue,
    }
    if instance_features is not None:
        payload["instance_features"] = instance_features
    return json.dumps(payload, ensure_ascii=False, indent=2)
