"""Language model router variants 1-3 of PROTOCOL.md section 5, on the clean
validation catalogue:

  1. Original-prompt router          - pilot prompt including the rule "select
                                       the evidence verifier for entity-specific,
                                       numerical, temporal, or externally
                                       checkable claims".
  2. Router without the evidence-verifier rule
                                     - the current prompt (Table 6's proposed router).
  3. Router without the rule and without retrieved-example metadata
                                     - current prompt, no retrieved examples.

Every variant is run on the validation partition (to choose a variant without
looking at test) and on the test partition (for reporting). Validation items
are in the train+validation retrieval pool, so each one excludes itself and
its near-duplicates from its own retrieved examples.

Every router call is cached per item (resumable after a power cut). The test
caches are shared with Table 6 (variant 2, "router") and Table 10 (variant 3,
"router_no_examples").

Usage: python -m experiments.run_router_variants --config configs/default.yaml
"""
from __future__ import annotations

from domain.predict import predict_domain_batch
from eval.metrics import macro_f1, veracity_report
from experiments.common import build_catalogue_for, build_encoders, build_registry, build_routing_domain_classifier, cache_path_for, load_and_split_amtcele, load_config, parse_args, prediction_cache_dir, split_by
from retrieval.index import build_index_for_pool
from routing.llm_router import LLMRouter
from routing.oracle import correct_tools_per_item, outputs_from_decisions
from routing.selector_prompt import ORIGINAL_SELECTOR_SYSTEM_PROMPT, SELECTOR_SYSTEM_PROMPT
from utils.cache import cached_map
from utils.io import write_json
from utils.types import RouterDecision, TOOL_IDS

# (display name, cache name, system prompt, use retrieved examples)
VARIANTS = [
    ("Original-prompt router", "router_original_prompt", ORIGINAL_SELECTOR_SYSTEM_PROMPT, True),
    ("Router without evidence-verifier rule", "router", SELECTOR_SYSTEM_PROMPT, True),
    ("Router without evidence-verifier rule or retrieved-example metadata", "router_no_examples", SELECTOR_SYSTEM_PROMPT, False),
]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    train_items, val_items, test_items = split_by(items, "train"), split_by(items, "val"), split_by(items, "test")

    registry = build_registry(cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed, t1_checkpoint_key=f"seed{seed}")
    cache_dir = prediction_cache_dir(cfg, seed)
    catalogue, val_outputs_by_tool = build_catalogue_for(registry, val_items, cache_dir=cache_dir)
    test_outputs_by_tool = {
        tid: registry.get(tid).predict_batch(test_items, cache_path=cache_path_for(cache_dir, tid, "test"))
        for tid in registry.all_ids()
    }

    domain_model = build_routing_domain_classifier(cfg, train_items, val_items, seed, checkpoint_key=f"seed{seed}")
    index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)

    partitions = {}
    for name, part_items, outputs_by_tool in (("val", val_items, val_outputs_by_tool), ("test", test_items, test_outputs_by_tool)):
        texts = [it.text for it in part_items]
        sem, aff = semantic_encoder.encode(texts), affective_encoder.encode(texts)
        exclude = [[it.id] for it in part_items] if name == "val" else None
        partitions[name] = {
            "items": part_items,
            "outputs_by_tool": outputs_by_tool,
            "domain_preds": predict_domain_batch(domain_model, texts),
            "affective": aff,
            "retrieved": index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"], exclude_ids_per_query=exclude),
            "correct_sets": correct_tools_per_item(part_items, outputs_by_tool),
        }
    if any(r.item.id == it.id for it, rs in zip(val_items, partitions["val"]["retrieved"]) for r in rs):
        raise RuntimeError("a validation item retrieved itself")

    results = {"val": [], "test": []}
    item_predictions = {"item_ids": [it.id for it in test_items], "gold": [it.label for it in test_items], "predictions": {}, "selected_tool": {}}
    for display, cache_name, prompt, use_examples in VARIANTS:
        router = LLMRouter(
            llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
            validation_catalogue=catalogue, mode=cfg["router"]["mode"],
            max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
            system_prompt=prompt,
        )
        for part_name, p in partitions.items():
            part_items = p["items"]

            def _select(i: int, p=p, part_items=part_items) -> RouterDecision:
                return router.select(
                    part_items[i], p["domain_preds"][i],
                    affective_summary={affective_encoder.labels[j]: float(p["affective"][i][j]) for j in range(len(affective_encoder.labels))},
                    retrieved_examples=p["retrieved"][i] if use_examples else [],
                    router_name=cache_name,
                )

            decisions = cached_map(
                [it.id for it in part_items], _select, lambda d: RouterDecision(**d),
                cache_path=cache_path_for(cache_dir, cache_name, part_name), label=f"{cache_name}:{part_name}",
            )
            outputs = outputs_from_decisions(decisions, p["outputs_by_tool"])
            gold = [it.label for it in part_items]
            selected = [d.selected_tool for d in decisions]
            results[part_name].append({
                "variant": display,
                **veracity_report(gold, [o.label for o in outputs], [o.prob_fake for o in outputs]),
                "routing_accuracy": sum(s in cs for s, cs in zip(selected, p["correct_sets"])) / len(selected),
                "tool_distribution": {t: selected.count(t) for t in TOOL_IDS},
                "fallbacks": sum(d.short_justification.startswith("Fallback") for d in decisions),
                "mean_router_seconds": sum(d.latency_s for d in decisions) / len(decisions),
            })
            if part_name == "test":
                item_predictions["predictions"][display] = {"label": [o.label for o in outputs], "prob_fake": [o.prob_fake for o in outputs]}
                item_predictions["selected_tool"][display] = selected

    # Choose on validation macro F1; ties go to the variant listed first in the protocol.
    best_val = max(results["val"], key=lambda r: r["macro_f1"])
    out_dir = cfg["paths"]["output_dir"]
    write_json(f"{out_dir}/router_variants.json", {
        "selected_on_validation": best_val["variant"],
        "validation": results["val"],
        "test": results["test"],
        "note": "Validation routing uses the catalogue computed on the same validation items; it is used only to compare variants.",
    })
    write_json(f"{out_dir}/router_variants_item_predictions.json", item_predictions)

    for part_name in ("val", "test"):
        print(f"\n--- {part_name} ---")
        for r in results[part_name]:
            print(f"{r['variant']:70s} | Macro-F1={r['macro_f1']:.3f} | routing acc={r['routing_accuracy']:.3f} | "
                  f"tools={r['tool_distribution']} | fallbacks={r['fallbacks']}")
    print(f"\nChosen on validation: {best_val['variant']}")


if __name__ == "__main__":
    main()
