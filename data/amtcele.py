"""Loader for AMTCele = FakeNewsAMT (6 domains) + Celebrity dataset (Perez-Rosas et al. 2018) [4].

Two raw layouts are supported:

1. The *actual* public release layout distributed as `fakeNewsDatasets.zip`
   from http://lit.eecs.umich.edu/downloads.html (verified against the real
   archive: `fake`/`legit` are the top-level folders, not domain; domain is
   encoded in the FakeNewsAMT filename prefix, and Celebrity is a sibling
   directory with no domain code since it's a single domain):

       raw_dir/
         fakeNewsDataset/
           fake/<code><NN>.fake.txt    e.g. biz01.fake.txt
           legit/<code><NN>.legit.txt  e.g. biz01.legit.txt
         celebrityDataset/
           fake/<NNN>fake.txt          e.g. 001fake.txt
           legit/<NNN>legit.txt        e.g. 001legit.txt

   where <code> in {biz, edu, entmt, polit, sports, tech} maps to
   {business, education, entertainment, politics, sports, technology}.

2. A generic domain-folder layout, for users who have already reorganised
   the corpus themselves (or a differently-mirrored copy):

       raw_dir/
         technology/{Fake,Legit}/*.txt
         education/{Fake,Legit}/*.txt
         ...
         celebrity/{fake,legit}/*.txt

   Folder/casing is tolerant of "Legit"/"Legitimate"/"real" and
   "Fake"/"fake_content" naming.

A flat CSV manifest (columns: text,label,domain[,id]) is also supported for
users who have already consolidated the corpus themselves.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional

from utils.types import DOMAINS, NewsItem

_FAKE_DIR_RE = re.compile(r"fake", re.IGNORECASE)
_LEGIT_DIR_RE = re.compile(r"legit|real|true", re.IGNORECASE)

_CELEBRITY_ALIAS = {"celebrity", "celebrities", "gossipcop", "gossip"}

# FakeNewsAMT filename domain-code prefixes, e.g. "biz01.fake.txt" -> business.
_DOMAIN_CODE_MAP = {
    "biz": "business",
    "edu": "education",
    "entmt": "entertainment",
    "polit": "politics",
    "sports": "sports",
    "tech": "technology",
}
_FILENAME_CODE_RE = re.compile(r"^([a-z]+)\d+", re.IGNORECASE)


def _normalise_domain(name: str) -> str:
    name = name.strip().lower()
    if name in _CELEBRITY_ALIAS:
        return "celebrity"
    return name


def _domain_from_filename(filename: str) -> Optional[str]:
    match = _FILENAME_CODE_RE.match(filename)
    if not match:
        return None
    return _DOMAIN_CODE_MAP.get(match.group(1).lower())


def _load_release_subdir(fake_legit_dir: Path, domain: Optional[str], counter: int, items: List[NewsItem]) -> int:
    """Reads one `{fake,legit}/` pair under either `fakeNewsDataset/` or
    `celebrityDataset/`. If `domain` is None (FakeNewsAMT), the domain is
    parsed per-file from the filename code prefix; otherwise (Celebrity)
    every file in this tree uses the fixed `domain`."""
    for label_name, label in (("fake", 1), ("legit", 0)):
        label_dir = fake_legit_dir / label_name
        if not label_dir.is_dir():
            continue
        for txt_file in sorted(label_dir.glob("*.txt")):
            resolved_domain = domain or _domain_from_filename(txt_file.name)
            if resolved_domain is None or resolved_domain not in DOMAINS:
                continue
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            items.append(
                NewsItem(
                    id=f"amtcele-{counter}",
                    text=text,
                    label=label,
                    domain=resolved_domain,
                    dataset="amtcele",
                    meta={"source_file": str(txt_file)},
                )
            )
            counter += 1
    return counter


def load_amtcele_from_release_layout(raw_dir: str | Path) -> List[NewsItem]:
    """Loads the exact public release archive layout (see module docstring,
    format 1). `raw_dir` is the folder containing `fakeNewsDataset/` and/or
    `celebrityDataset/` (i.e. the extracted `fakeNewsDatasets/` root)."""
    raw_dir = Path(raw_dir)
    items: List[NewsItem] = []
    counter = 0

    fnamt_dir = raw_dir / "fakeNewsDataset"
    if fnamt_dir.is_dir():
        counter = _load_release_subdir(fnamt_dir, domain=None, counter=counter, items=items)

    celeb_dir = raw_dir / "celebrityDataset"
    if celeb_dir.is_dir():
        counter = _load_release_subdir(celeb_dir, domain="celebrity", counter=counter, items=items)

    if not items:
        raise FileNotFoundError(
            f"No AMTCele items found under {raw_dir} using the release layout "
            "(expected fakeNewsDataset/ and/or celebrityDataset/ subfolders each "
            "containing fake/ and legit/ *.txt files)."
        )
    return items


def load_amtcele_from_dirs(raw_dir: str | Path) -> List[NewsItem]:
    """Scan the generic per-domain Fake/Legit directory layout (format 2)."""
    raw_dir = Path(raw_dir)
    items: List[NewsItem] = []
    counter = 0

    for domain_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        domain = _normalise_domain(domain_dir.name)
        if domain not in DOMAINS:
            continue
        for label_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            if _FAKE_DIR_RE.search(label_dir.name):
                label = 1
            elif _LEGIT_DIR_RE.search(label_dir.name):
                label = 0
            else:
                continue
            for txt_file in sorted(label_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                items.append(
                    NewsItem(
                        id=f"amtcele-{counter}",
                        text=text,
                        label=label,
                        domain=domain,
                        dataset="amtcele",
                        meta={"source_file": str(txt_file)},
                    )
                )
                counter += 1
    if not items:
        raise FileNotFoundError(
            f"No AMTCele items found under {raw_dir}. Expected "
            f"<domain>/{{Fake,Legit}}/*.txt subfolders for domains in {DOMAINS}, "
            "or the release layout (fakeNewsDataset/ + celebrityDataset/)."
        )
    return items


def load_amtcele_from_csv(csv_path: str | Path) -> List[NewsItem]:
    csv_path = Path(csv_path)
    items: List[NewsItem] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            domain = _normalise_domain(row["domain"])
            if domain not in DOMAINS:
                raise ValueError(f"Unknown AMTCele domain {row['domain']!r} at row {i}")
            items.append(
                NewsItem(
                    id=str(row.get("id", f"amtcele-{i}")),
                    text=row["text"],
                    label=int(row["label"]),
                    domain=domain,
                    dataset="amtcele",
                )
            )
    return items


def load_amtcele(path: str | Path) -> List[NewsItem]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return load_amtcele_from_csv(path)
    if (path / "fakeNewsDataset").is_dir() or (path / "celebrityDataset").is_dir():
        return load_amtcele_from_release_layout(path)
    return load_amtcele_from_dirs(path)
