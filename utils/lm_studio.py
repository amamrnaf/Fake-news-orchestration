"""Load/unload LM Studio's model via its `lms` CLI, so local GPU training
(T1, domain classifier) can get full VRAM headroom without permanently
giving up the LLM for the rest of a script's run. Confirmed empirically:
unload is near-instant, reload takes ~5s and comes back fully responsive.

No-ops (with a warning, not a crash) if `lms` isn't on PATH, so scripts using
this stay runnable in environments without LM Studio's CLI installed.
"""
from __future__ import annotations

import shutil
import subprocess

_LMS = shutil.which("lms")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    # lms's loading-spinner output uses Unicode braille characters that
    # aren't representable in Windows' default console codepage (cp1252);
    # decoding explicitly as UTF-8 with errors replaced avoids the capture
    # thread crashing on that output (the command itself still succeeds
    # either way, but the crash is noisy and worth avoiding in an
    # unattended multi-hour run).
    return subprocess.run(args, check=False, capture_output=True, encoding="utf-8", errors="replace")


def unload_llm(model_id: str) -> None:
    if not _LMS:
        print("[lm_studio] 'lms' CLI not found on PATH -- skipping unload", flush=True)
        return
    print(f"[lm_studio] unloading {model_id} to free VRAM for local training", flush=True)
    _run([_LMS, "unload", model_id])


def load_llm(model_id: str) -> None:
    if not _LMS:
        print("[lm_studio] 'lms' CLI not found on PATH -- skipping reload", flush=True)
        return
    print(f"[lm_studio] reloading {model_id}", flush=True)
    result = _run([_LMS, "load", model_id, "-y"])
    if result.returncode != 0:
        print(f"[lm_studio] WARNING: reload may have failed: {result.stderr or result.stdout}", flush=True)
