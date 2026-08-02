"""Builds data_raw/evidence_corpus.jsonl for T3 from Wikipedia summaries of
named entities extracted from the AMTCele corpus (per-domain top entities,
so coverage isn't dominated by the oversized Celebrity partition).

Usage: python -m scripts.build_evidence_corpus --config configs/default.yaml

Source: Wikipedia REST API summary endpoint (lead section of each article).
Content is CC BY-SA 4.0 / GFDL (https://en.wikipedia.org/wiki/Wikipedia:Copyrights) —
cite Wikipedia as the evidence source if this corpus is described in the paper.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import requests
import spacy
import yaml

from data.amtcele import load_amtcele

KEEP_LABELS = {"PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW"}
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
USER_AGENT = "FakeNewsOrchestrationResearch/1.0 (evidence corpus for T3; contact: https://github.com/amamrnaf/Fake-news-orchestration)"


def extract_entities(amtcele_dir: str, per_domain_top_k: int = 80, min_count: int = 2) -> list[str]:
    items = load_amtcele(amtcele_dir)
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    by_domain = defaultdict(Counter)
    for item in items:
        doc = nlp(item.text[:2000])
        for ent in doc.ents:
            t = ent.text.strip()
            if ent.label_ in KEEP_LABELS and len(t) > 2 and not t.isdigit():
                by_domain[item.domain][t] += 1

    all_entities: list[str] = []
    for counter in by_domain.values():
        top = [e for e, c in counter.most_common(per_domain_top_k) if c >= min_count]
        all_entities.extend(top)
    return list(dict.fromkeys(all_entities))


def fetch_summary(title: str, session: requests.Session) -> dict | None:
    url = WIKI_API.format(urllib.parse.quote(title.replace(" ", "_")))
    try:
        resp = session.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("type") == "disambiguation":
        return None
    extract = data.get("extract", "").strip()
    if not extract:
        return None
    return {"title": data.get("title", title), "extract": extract}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_path = Path(cfg["paths"]["evidence_corpus"])

    print("Extracting entities from AMTCele...")
    entities = extract_entities(cfg["paths"]["amtcele_raw"])
    print(f"{len(entities)} unique entities to look up on Wikipedia")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    passages = []
    misses = 0
    seen_titles = set()
    for i, entity in enumerate(entities):
        result = fetch_summary(entity, session)
        if result and result["title"] not in seen_titles:
            seen_titles.add(result["title"])
            passages.append({"doc_id": f"wiki-{len(passages):04d}", "text": f"{result['title']}: {result['extract']}"})
        else:
            misses += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(entities)} looked up, {len(passages)} passages so far")
        time.sleep(args.sleep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(passages)} passages to {out_path} ({misses} entities had no usable page)")


if __name__ == "__main__":
    main()
