"""Produces Table 8: tool-selection quality (random / domain-rule / LLM
domain-only / LLM full-context routers) plus the selection confusion matrix
(rows = oracle/best tool, columns = selected tool) requested after Table 8.

Usage: python -m experiments.run_routing_analysis --config configs/default.yaml
"""
from __future__ import annotations

from domain.predict import predict_domain_batch
from eval.routing_metrics import (
    mean_selection_regret,
    oracle_gap,
    selection_confusion_matrix,
    tool_selection_distribution,
)
from eval.metrics import macro_f1
from experiments.common import build_catalogue_for, build_encoders, build_registry, build_routing_domain_classifier, load_and_split_amtcele, load_config, parse_args, split_by
from retrieval.index import build_index_for_pool
from routing.baselines import DomainRuleRouter, RandomRouter
from routing.llm_router import LLMRouter
from routing.oracle import correct_tools_per_item, oracle_select, outputs_from_decisions
from utils.io import write_json
from utils.types import TOOL_IDS


def _route_and_score(name, decisions, outputs_by_tool, test_items, gold, oracle_f1, correct_sets):
    outputs = outputs_from_decisions(decisions, outputs_by_tool)
    router_f1 = macro_f1(gold, [o.label for o in outputs])
    selected = [d.selected_tool for d in decisions]
    best_labels = [next(iter(sorted(cs))) if cs else "T1" for cs in correct_sets]
    return {
        "router": name,
        "selection_accuracy": float(sum(s in cs for s, cs in zip(selected, correct_sets)) / len(selected)),
        "oracle_gap": oracle_gap(router_f1, oracle_f1),
        "mean_regret": mean_selection_regret(
            [{tid: (test_items[i].label == outputs_by_tool[tid][i].label) for tid in TOOL_IDS} for i in range(len(test_items))],
            selected,
        ),
        "llm_calls": sum(d.llm_calls for d in decisions) / len(decisions),
        "tool_distribution": tool_selection_distribution(selected),
        "_selected": selected,
        "_best_labels": best_labels,
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

    oracle_decisions = oracle_select(test_items, outputs_by_tool)
    oracle_f1 = macro_f1(gold, [o.label for o in outputs_from_decisions(oracle_decisions, outputs_by_tool)])

    domain_model = build_routing_domain_classifier(cfg, train_items, val_items, seed, checkpoint_key=f"seed{seed}")
    domain_preds = predict_domain_batch(domain_model, [it.text for it in test_items])

    index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)
    sem = semantic_encoder.encode([it.text for it in test_items])
    aff = affective_encoder.encode([it.text for it in test_items])
    retrieved = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"])
    aff_summary = affective_encoder.encode([it.text for it in test_items])

    random_router = RandomRouter(seed=seed)
    random_decisions = [random_router.select(it) for it in test_items]

    domain_rule = DomainRuleRouter(catalogue)
    dr_decisions = [domain_rule.select(test_items[i], domain_preds[i]) for i in range(len(test_items))]

    llm_domain_only = LLMRouter(
        llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
        validation_catalogue=catalogue, mode="domain_catalogue",
        max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
    )
    llm_domain_decisions = [
        llm_domain_only.select(
            test_items[i], domain_preds[i],
            affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
            retrieved_examples=retrieved[i], router_name="llm_domain_only",
        )
        for i in range(len(test_items))
    ]

    llm_full_context = LLMRouter(
        llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
        validation_catalogue=catalogue, mode="instance_aware",
        max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
    )
    llm_full_decisions = [
        llm_full_context.select(
            test_items[i], domain_preds[i],
            affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
            retrieved_examples=retrieved[i], router_name="llm_full_context",
        )
        for i in range(len(test_items))
    ]

    rows = [
        _route_and_score("Random", random_decisions, outputs_by_tool, test_items, gold, oracle_f1, correct_sets),
        _route_and_score("Domain rule", dr_decisions, outputs_by_tool, test_items, gold, oracle_f1, correct_sets),
        _route_and_score("LLM, domain only", llm_domain_decisions, outputs_by_tool, test_items, gold, oracle_f1, correct_sets),
        _route_and_score("LLM, full context", llm_full_decisions, outputs_by_tool, test_items, gold, oracle_f1, correct_sets),
    ]

    best_full_context = rows[-1]
    confusion = selection_confusion_matrix(best_full_context["_best_labels"], best_full_context["_selected"])

    for row in rows:
        row.pop("_selected")
        row.pop("_best_labels")

    write_json(f"{cfg['paths']['output_dir']}/table8_routing_analysis.json", rows)
    write_json(
        f"{cfg['paths']['output_dir']}/table8_selection_confusion_matrix.json",
        {"labels": TOOL_IDS, "matrix": confusion.tolist(), "note": "rows=oracle/best tool, columns=selected tool (LLM full-context router)"},
    )

    for row in rows:
        print(
            f"{row['router']:20s} | sel.acc={row['selection_accuracy']:.3f} | oracle_gap={row['oracle_gap']:.3f} | "
            f"regret={row['mean_regret']:.3f} | llm_calls={row['llm_calls']:.2f}"
        )


if __name__ == "__main__":
    main()
