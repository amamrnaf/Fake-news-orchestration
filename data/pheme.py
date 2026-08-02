"""Loader for PHEME (Kochkina et al. 2018 [5]) — leave-one-event-out rumour verification.

Expected raw layout (the standard `pheme-rnr-dataset` / `PHEME_veracity` release):

    raw_dir/
      <event-name>/
        rumours/<thread_id>/source-tweet/<thread_id>.json
        non-rumours/<thread_id>/source-tweet/<thread_id>.json

`event-name` (e.g. charliehebdo, ferguson, germanwings-crash, ottawashooting,
sydneysiege, ...) becomes the leave-one-event-out grouping key. Each thread's
own id is stored as the near-duplicate/thread grouping key so reactions from
the same conversation never cross a split (Sec 4.2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from utils.types import NewsItem


def _read_source_tweet_text(thread_dir: Path) -> str | None:
    source_dir = thread_dir / "source-tweet"
    if not source_dir.is_dir():
        return None
    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        return None
    with open(json_files[0], "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("text") or payload.get("full_text")


def load_pheme(raw_dir: str | Path) -> List[NewsItem]:
    raw_dir = Path(raw_dir)
    items: List[NewsItem] = []

    for event_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        event = event_dir.name
        for label_name, label in (("rumours", 1), ("non-rumours", 0)):
            label_dir = event_dir / label_name
            if not label_dir.is_dir():
                continue
            for thread_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                text = _read_source_tweet_text(thread_dir)
                if not text:
                    continue
                items.append(
                    NewsItem(
                        id=f"pheme-{event}-{thread_dir.name}",
                        text=text,
                        label=label,
                        domain=None,
                        dataset="pheme",
                        group=event,  # leave-one-event-out key
                        meta={"event": event, "thread_id": thread_dir.name},
                    )
                )
    if not items:
        raise FileNotFoundError(
            f"No PHEME items found under {raw_dir}. Expected "
            "<event>/{rumours,non-rumours}/<thread_id>/source-tweet/*.json"
        )
    return items
