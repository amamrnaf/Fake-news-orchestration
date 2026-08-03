"""Shared setup used by every experiments/run_*.py script: load config, load
and split AMTCele, build encoders/retrieval index, build the detector
registry, and build the validation catalogue. Kept in one place so every
results table (Sec 5-6) is computed from the same protocol, per Sec 3.4:
'so that the performance catalogue reflects a common protocol rather than
incomparable scores'.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

from perf_catalogue.build_catalogue import build_validation_catalogue, run_detectors_on_validation
from data.amtcele import load_amtcele
from data.splits import stratified_split
from detectors.llm_backend import build_llm_backend
from detectors.registry import DetectorRegistry
from detectors.t1_supervised import SupervisedTransformerDetector, T1Config
from detectors.t2_retrieval_llm import RetrievedExampleLLMDetector
from detectors.t3_evidence_verifier import EvidenceCorpus, EvidenceGroundedVerifier, EvidencePassage
from domain.model import DomainClassifierModel
from domain.train import TrainConfig, train_one_config
from retrieval.embeddings import AffectiveEncoder, SemanticEncoder
from retrieval.index import build_index_for_pool
from utils.io import load_yaml, read_jsonl
from utils.types import NewsItem


def parse_args(default_config: str = "configs/default.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seed", type=int, default=None, help="Override configs/seeds[0].")
    return parser.parse_args()


def load_config(path: str) -> dict:
    return load_yaml(path)


def load_and_split_amtcele(cfg: dict, seed: int, semantic_encoder=None) -> List[NewsItem]:
    items = load_amtcele(cfg["paths"]["amtcele_raw"])
    return stratified_split(
        items,
        train_size=cfg["splits"]["train_size"],
        val_size=cfg["splits"]["val_size"],
        test_size=cfg["splits"]["test_size"],
        seed=seed,
        stratify_by_domain=True,
        semantic_encoder=semantic_encoder,
    )


def split_by(items: Sequence[NewsItem], split: str) -> List[NewsItem]:
    return [it for it in items if it.split == split]


def build_encoders(cfg: dict):
    semantic_encoder = SemanticEncoder(cfg["retrieval"]["semantic_encoder"])
    affective_encoder = AffectiveEncoder(cfg["retrieval"]["affective_encoder"])
    return semantic_encoder, affective_encoder


def build_evidence_corpus(cfg: dict) -> EvidenceCorpus:
    passages = [EvidencePassage(doc_id=row["doc_id"], text=row["text"]) for row in read_jsonl(cfg["paths"]["evidence_corpus"])]
    return EvidenceCorpus(passages)


def build_routing_domain_classifier(
    cfg: dict,
    train_items: Sequence[NewsItem],
    val_items: Sequence[NewsItem],
    seed: int,
    checkpoint_key: Optional[str] = None,
) -> DomainClassifierModel:
    """The lightweight, single-config domain classifier used by
    run_main_veracity/run_ablations/run_routing_analysis/run_efficiency purely
    as routing context (NOT the full 3-encoder x 12-config grid search from
    run_domain_classification.py / Table 5 -- that one's result is never reused
    elsewhere in the pipeline). Always trains `candidate_encoders[0]` with the
    first value of each hyperparameter grid.

    `checkpoint_key` (e.g. f"seed{seed}") opts into caching under
    cfg["paths"]["quick_domain_classifier_checkpoint_dir"], the same pattern as
    `build_registry`'s `t1_checkpoint_key`. Leave None for
    run_leave_one_domain_out, which uses a different split per fold."""
    checkpoint_dir = cfg["paths"].get("quick_domain_classifier_checkpoint_dir")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_key) if checkpoint_dir and checkpoint_key else None

    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "wrapper_meta.json")):
        print(f"[domain classifier] loaded cached checkpoint from {checkpoint_path}, skipping training", flush=True)
        return DomainClassifierModel.load(checkpoint_path)

    dc_cfg = cfg["domain_classifier"]
    dom_cfg = TrainConfig(
        model_name=dc_cfg["candidate_encoders"][0],
        lr=dc_cfg["lr_grid"][0],
        batch_size=dc_cfg["batch_size_grid"][0],
        max_length=dc_cfg["max_length_grid"][0],
        seed=seed,
    )
    domain_model, _ = train_one_config(train_items, val_items, dom_cfg)
    if checkpoint_path:
        domain_model.save(checkpoint_path)
        print(f"[domain classifier] saved checkpoint to {checkpoint_path}", flush=True)
    return domain_model


def build_registry(
    cfg: dict,
    train_items: Sequence[NewsItem],
    val_items: Sequence[NewsItem],
    retrieval_pool: Sequence[NewsItem],
    semantic_encoder,
    affective_encoder,
    seed: int,
    t1_checkpoint_key: Optional[str] = None,
) -> DetectorRegistry:
    """Trains T1, and wires T2/T3 to a shared LLM backend + retrieval index
    built only from `retrieval_pool` (must exclude the target/held-out
    partition, Sec 3.2 leakage control).

    `t1_checkpoint_key` (e.g. f"seed{seed}") opts into loading/saving T1 under
    `cfg["paths"]["t1_checkpoint_dir"]`, letting it be trained once (e.g. on a
    Colab GPU) and reused across every script that shares the same train/val
    split for that seed (run_catalogue, run_main_veracity, run_ablations,
    run_routing_analysis, run_efficiency all do). Leave None (e.g. for LODO,
    which uses a different split per fold) to always train fresh."""
    llm = build_llm_backend(cfg["llm"])

    t1_checkpoint_dir = cfg["paths"].get("t1_checkpoint_dir")
    checkpoint_path = (
        os.path.join(t1_checkpoint_dir, t1_checkpoint_key)
        if t1_checkpoint_dir and t1_checkpoint_key
        else None
    )
    if checkpoint_path and SupervisedTransformerDetector.has_checkpoint(checkpoint_path):
        print(f"[T1] loaded cached checkpoint from {checkpoint_path}, skipping training", flush=True)
        t1 = SupervisedTransformerDetector.load(checkpoint_path)
    else:
        t1_cfg = T1Config(**{**cfg["t1_supervised"], "seed": seed})
        t1 = SupervisedTransformerDetector(t1_cfg)
        t1.fit(train_items, val_items)
        if checkpoint_path:
            t1.save(checkpoint_path)
            print(f"[T1] saved checkpoint to {checkpoint_path}", flush=True)

    index = build_index_for_pool(retrieval_pool, semantic_encoder, affective_encoder)
    t2 = RetrievedExampleLLMDetector(
        llm=llm,
        index=index,
        semantic_encoder=semantic_encoder,
        affective_encoder=affective_encoder,
        mode="affective_semantic",
        k=cfg["retrieval"]["k"],
        alpha=cfg["retrieval"]["alpha"],
        max_new_tokens=cfg["llm"]["max_new_tokens"],
        seed=seed,
    )

    corpus = build_evidence_corpus(cfg)
    t3 = EvidenceGroundedVerifier(
        llm=llm,
        corpus=corpus,
        k=cfg["t3_evidence_verifier"]["bm25_k"],
        min_score=cfg["t3_evidence_verifier"]["min_score"],
        max_new_tokens=cfg["llm"]["max_new_tokens"],
    )

    return DetectorRegistry(t1, t2, t3)


def build_catalogue_for(
    registry: DetectorRegistry, val_items: Sequence[NewsItem], cache_dir: Optional[str] = None
) -> dict:
    outputs_by_tool = run_detectors_on_validation(registry, val_items, cache_dir=cache_dir)
    return build_validation_catalogue(val_items, outputs_by_tool), outputs_by_tool
