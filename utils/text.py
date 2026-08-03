"""Shared text-length safety helpers.

Prompts built elsewhere embed full article/claim text with no cap, which can
exceed a locally-served model's context window (observed: a 6332-token T2
prompt against a 4096-token LM Studio context, crashing the whole run). This
is a rough character-based truncation (no tokenizer dependency), sized
generously below typical context windows.
"""
from __future__ import annotations

DEFAULT_MAX_CHARS = 4000


def truncate_chars(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [...truncated]"
