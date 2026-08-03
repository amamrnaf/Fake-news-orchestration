"""Produces Table 10: ablation study isolating affective retrieval, retrieved
examples, domain information, and the validation catalogue (Sec 6).

Configurations:
  Complete framework            - full instance-aware router, tuned alpha/k.
  Without affective vector      - alpha=1 (semantic-only retrieval).
  Without retrieved examples    - k=0 (no top-k example metadata).
  Without domain distribution   - router sees only the top-1 label, not the
                                   full distribution/entropy (one-hot, entropy=0).
  Hard domain label only        - same as above (kept distinct in the table
                                   per the paper; here both realise "hard
                                   label" since the distribution is the only
                                   soft signal the router receives).
  No validation catalogue       - empty catalogue passed to the router.
  Gold domain                   - predicted_domain replaced by the gold label
                                   with confidence 1.0, entropy 0.0.
  All tools for every input     - confidence-weighted ensemble of T1+T2+T3
                                   (tools/input fixed at 3.00, Table 10).

Usage: python -m experiments.run_ablations --config configs/default.yaml
"""
from __future__ import annotations

import copy
from dataclasses import replace

from domain.predict import predict_domain_batch
from eval.metrics import expected_calibration_error, macro_f1
from experiments.common import build_catalogue_for, build_encoders, build_registry, build_routing_domain_classifier, load_and_split_amtcele, load_config, parse_args, split_by
from retrieval.index import build_index_for_pool
from routing.baselines import confidence_weighted_ensemble
from routing.llm_router import LLMRouter
from routing.oracle import correct_tools_per_item, outputs_from_decisions
from utils.io import write_json
from utils.types import DomainPrediction


def _one_hot_domain_pred(dp: DomainPrediction) -> DomainPrediction:
    dist = {d: (1.0 if d == dp.predicted_domain else 0.0) for d in dp.distribution}
    return DomainPrediction(distribution=dist, predicted_domain=dp.predicted_domain, entropy=0.0, confidence=1.0)


def _gold_domain_pred(gold_domain: str, template: DomainPrediction) -> DomainPrediction:
    dist = {d: (1.0 if d == gold_domain else 0.0) for d in template.distribution}
    return DomainPrediction(distribution=dist, predicted_domain=gold_domain, entropy=0.0, confidence=1.0)


def _run_router_config(name, cfg, test_items, gold, registry, catalogue, domain_preds, retrieved, affective_encoder, correct_sets, outputs_by_tool):
    aff_summary = affective_encoder.encode([it.text for it in test_items])
    router = LLMRouter(
        llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
        validation_catalogue=catalogue, mode=cfg["router"]["mode"],
        max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
    )
    decisions = [
        router.select(
            test_items[i], domain_preds[i],
            affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
            retrieved_examples=retrieved[i],
        )
        for i in range(len(test_items))
    ]
    outputs = outputs_from_decisions(decisions, outputs_by_tool)
    y_pred = [o.label for o in outputs]
    y_conf = [d.selection_confidence for d in decisions]

    selected = [d.selected_tool for d in decisions]
    routing_accuracy = sum(s in cs for s, cs in zip(selected, correct_sets)) / len(selected)
    tools_per_input = sum(1 + d.llm_calls for d in decisions) / len(decisions)

    return {
        "configuration": name,
        "macro_f1": macro_f1(gold, y_pred),
        "routing_accuracy": routing_accuracy,
        "ece": expected_calibration_error(gold, y_pred, y_conf),
        "tools_per_input": tools_per_input,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    train_items, val_items, test_items = split_by(items, "train"), split_by(items, "val"), split_by(items, "test")
    gold = [it.label for it in test_items]

    registry = build_registry(cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed, t1_checkpoint_key=f"seed{seed}")
    catalogue, _ = build_catalogue_for(registry, val_items)
    outputs_by_tool = {tid: registry.get(tid).predict_batch(test_items) for tid in registry.all_ids()}
    correct_sets = correct_tools_per_item(test_items, outputs_by_tool)

    domain_model = build_routing_domain_classifier(cfg, train_items, val_items, seed, checkpoint_key=f"seed{seed}")
    domain_preds = predict_domain_batch(domain_model, [it.text for it in test_items])

    index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)
    sem = semantic_encoder.encode([it.text for it in test_items])
    aff = affective_encoder.encode([it.text for it in test_items])

    retrieved_full = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"])
    retrieved_semantic_only = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=1.0)
    retrieved_none = [[] for _ in test_items]

    one_hot_preds = [_one_hot_domain_pred(dp) for dp in domain_preds]
    gold_domain_preds = [_gold_domain_pred(test_items[i].domain, domain_preds[i]) for i in range(len(test_items))]
    empty_catalogue = {}

    rows = [
        _run_router_config("Complete framework", cfg, test_items, gold, registry, catalogue, domain_preds, retrieved_full, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("Without affective vector", cfg, test_items, gold, registry, catalogue, domain_preds, retrieved_semantic_only, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("Without retrieved examples", cfg, test_items, gold, registry, catalogue, domain_preds, retrieved_none, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("Without domain distribution", cfg, test_items, gold, registry, catalogue, one_hot_preds, retrieved_full, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("Hard domain label only", cfg, test_items, gold, registry, catalogue, one_hot_preds, retrieved_full, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("No validation catalogue", cfg, test_items, gold, registry, empty_catalogue, domain_preds, retrieved_full, affective_encoder, correct_sets, outputs_by_tool),
        _run_router_config("Gold domain", cfg, test_items, gold, registry, catalogue, gold_domain_preds, retrieved_full, affective_encoder, correct_sets, outputs_by_tool),
    ]

    all_tools_outputs = [
        confidence_weighted_ensemble({tid: outputs_by_tool[tid][i] for tid in registry.all_ids()}) for i in range(len(test_items))
    ]
    rows.append(
        {
            "configuration": "All tools for every input",
            "macro_f1": macro_f1(gold, [o.label for o in all_tools_outputs]),
            "routing_accuracy": None,
            "ece": expected_calibration_error(gold, [o.label for o in all_tools_outputs], [max(o.prob_fake, 1 - o.prob_fake) for o in all_tools_outputs]),
            "tools_per_input": 3.00,
        }
    )

    write_json(f"{cfg['paths']['output_dir']}/table10_ablations.json", rows)
    for row in rows:
        acc_str = f"{row['routing_accuracy']:.3f}" if row["routing_accuracy"] is not None else "-"
        print(
            f"{row['configuration']:32s} | macro-F1={row['macro_f1']:.3f} | routing_acc={acc_str} | "
            f"ECE={row['ece']:.3f} | tools/input={row['tools_per_input']:.2f}"
        )


if __name__ == "__main__":
    main()
