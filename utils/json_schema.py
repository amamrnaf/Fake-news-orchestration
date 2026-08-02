"""Strict parsing/validation for the router's JSON output (Appendix A schema).

The selector must return exactly:
{
  "selected_tool": "T1|T2|T3",
  "selection_confidence": 0.0,
  "reason_code": "DOMAIN_FIT|EVIDENCE_REQUIRED|OOD_REASONING|LOW_COST|OTHER",
  "short_justification": "..."
}
LLMs frequently wrap JSON in markdown fences or add trailing prose, so parsing
is defensive: extract the first balanced {...} block, then validate fields.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

VALID_TOOLS = {"T1", "T2", "T3"}
VALID_REASON_CODES = {"DOMAIN_FIT", "EVIDENCE_REQUIRED", "OOD_REASONING", "LOW_COST", "OTHER"}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class RouterSelection(BaseModel):
    selected_tool: str
    selection_confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str
    short_justification: str

    @field_validator("selected_tool")
    @classmethod
    def _check_tool(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_TOOLS:
            raise ValueError(f"selected_tool must be one of {VALID_TOOLS}, got {v!r}")
        return v

    @field_validator("reason_code")
    @classmethod
    def _check_reason(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_REASON_CODES:
            raise ValueError(f"reason_code must be one of {VALID_REASON_CODES}, got {v!r}")
        return v


class RouterSchemaError(ValueError):
    pass


def extract_json_block(raw_text: str) -> str:
    match = _JSON_OBJ_RE.search(raw_text)
    if not match:
        raise RouterSchemaError(f"No JSON object found in LLM output: {raw_text[:200]!r}")
    return match.group(0)


def parse_router_output(raw_text: str) -> RouterSelection:
    block = extract_json_block(raw_text)
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        raise RouterSchemaError(f"Malformed JSON from selector: {exc}") from exc
    try:
        return RouterSelection.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError
        raise RouterSchemaError(str(exc)) from exc


def try_parse_router_output(raw_text: str) -> Optional[RouterSelection]:
    try:
        return parse_router_output(raw_text)
    except RouterSchemaError:
        return None
