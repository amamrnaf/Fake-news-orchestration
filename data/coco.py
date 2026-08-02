"""Loader for COCO — COVID-19 Conspiracy Twitter dataset (Langguth et al. 2023 [6]).

The original release has 3,495 tweets with per-topic stance labels across twelve
conspiracy topics. The paper's "processed detection setting" (Sec 4.1) removes
instances unrelated to any topic and maps stance to a binary conspiracy/related
label. We expect a CSV export with (at minimum) tweet id/text, topic, and stance
columns; exact column names vary across releases of the dataset, so they are
configurable.

"Unrelated" stance values are dropped, matching the paper's processed setting.
Held-out topic groups (Table 3, "Held-out topic groups" protocol) use `topic`
as the grouping key for `data.splits.leave_one_group_out`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from utils.types import NewsItem

# Stance values considered "conspiracy-endorsing" -> label 1; everything else
# that isn't explicitly unrelated -> label 0 ("related" / non-endorsing).
_CONSPIRACY_STANCES = {"support", "supporting", "endorse", "endorsing", "agree", "conspiracy"}
_UNRELATED_STANCES = {"unrelated", "not_applicable", "n/a", "none", ""}


def load_coco(
    csv_path: str | Path,
    text_col: str = "text",
    topic_col: str = "topic",
    stance_col: str = "stance",
    id_col: Optional[str] = "tweet_id",
) -> List[NewsItem]:
    csv_path = Path(csv_path)
    items: List[NewsItem] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            stance = str(row.get(stance_col, "")).strip().lower()
            if stance in _UNRELATED_STANCES:
                continue  # processed detection setting removes unrelated instances
            label = 1 if stance in _CONSPIRACY_STANCES else 0
            text = row[text_col]
            topic = row[topic_col].strip()
            item_id = str(row[id_col]) if id_col and row.get(id_col) else f"coco-{i}"
            items.append(
                NewsItem(
                    id=f"coco-{item_id}",
                    text=text,
                    label=label,
                    domain=None,
                    dataset="coco",
                    group=topic,  # held-out topic groups
                    meta={"topic": topic, "raw_stance": stance},
                )
            )
    if not items:
        raise FileNotFoundError(f"No COCO items parsed from {csv_path}.")
    return items
