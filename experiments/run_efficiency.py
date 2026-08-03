"""Produces Table 11: efficiency comparison (seconds/item, generated tokens,
peak VRAM, relative cost vs. the all-tool advisor) on the local evaluation
machine (Sec 4.5). Also runs the secondary lambda sweep (Sec 3.1) tracing the
accuracy-cost trade-off of the closed-form utility router.

Usage: python -m experiments.run_efficiency --config configs/default.yaml
"""
from __future__ import annotations

from domain.predict import predict_domain_batch
from eval.efficiency import aggregate_efficiency, record_from_outputs, relative_cost
from eval.metrics import macro_f1
from experiments.common import build_catalogue_for, build_encoders, build_registry, build_routing_domain_classifier, load_and_split_amtcele, load_config, parse_args, split_by
from routing.baselines import AllToolLLMAdvisor, confidence_weighted_ensemble
from routing.llm_router import LLMRouter
from routing.oracle import outputs_from_decisions
from routing.utility_router import lambda_sweep
from retrieval.index import build_index_for_pool
from utils.io import write_json


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

    domain_model = build_routing_domain_classifier(cfg, train_items, val_items, seed, checkpoint_key=f"seed{seed}")
    domain_preds = predict_domain_batch(domain_model, [it.text for it in test_items])

    index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)
    sem = semantic_encoder.encode([it.text for it in test_items])
    aff = affective_encoder.encode([it.text for it in test_items])
    retrieved = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"])
    aff_summary = affective_encoder.encode([it.text for it in test_items])

    advisor = AllToolLLMAdvisor(llm=registry.get("T2").llm)
    advisor_outputs = [
        advisor.synthesise(test_items[i], {tid: outputs_by_tool[tid][i] for tid in registry.all_ids()})
        for i in range(len(test_items))
    ]
    advisor_records = [
        record_from_outputs(test_items[i].id, "advisor", [outputs_by_tool[tid][i] for tid in registry.all_ids()] + [advisor_outputs[i]])
        for i in range(len(test_items))
    ]

    router = LLMRouter(
        llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
        validation_catalogue=catalogue, mode=cfg["router"]["mode"],
        max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
    )
    proposed_decisions = [
        router.select(
            test_items[i], domain_preds[i],
            affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
            retrieved_examples=retrieved[i],
        )
        for i in range(len(test_items))
    ]
    proposed_outputs = outputs_from_decisions(proposed_decisions, outputs_by_tool)
    proposed_records = [
        record_from_outputs(test_items[i].id, "proposed", [proposed_outputs[i]], router_llm_calls=proposed_decisions[i].llm_calls)
        for i in range(len(test_items))
    ]

    t1_records = [record_from_outputs(it.id, "t1", [out]) for it, out in zip(test_items, outputs_by_tool["T1"])]
    t2_records = [record_from_outputs(it.id, "t2", [out]) for it, out in zip(test_items, outputs_by_tool["T2"])]
    t3_records = [record_from_outputs(it.id, "t3", [out]) for it, out in zip(test_items, outputs_by_tool["T3"])]

    summary = {
        "Supervised transformer": aggregate_efficiency(t1_records),
        "Retrieved-example LLM": aggregate_efficiency(t2_records),
        "Evidence verifier": aggregate_efficiency(t3_records),
        "All-tool advisor": aggregate_efficiency(advisor_records),
        "Proposed router": aggregate_efficiency(proposed_records),
    }

    reference = summary["All-tool advisor"]["seconds_per_item"]
    for method, stats in summary.items():
        stats["relative_cost"] = relative_cost(stats["seconds_per_item"], reference)

    write_json(f"{cfg['paths']['output_dir']}/table11_efficiency.json", summary)
    for method, stats in summary.items():
        print(
            f"{method:24s} | s/item={stats['seconds_per_item']:.3f} | tokens={stats['generated_tokens']:.1f} | "
            f"VRAM={stats['peak_vram_mb']:.0f}MB | rel.cost={stats['relative_cost']:.2f}"
        )

    # Secondary efficiency experiment: lambda sweep on the closed-form utility router (Sec 3.1).
    sweep = lambda_sweep(domain_preds, [it.id for it in test_items], catalogue, cfg["router"]["lambda_grid"])
    sweep_summary = []
    for lam, decisions in sweep.items():
        outputs = outputs_from_decisions(decisions, outputs_by_tool)
        sweep_summary.append({"lambda": lam, "macro_f1": macro_f1(gold, [o.label for o in outputs])})
    write_json(f"{cfg['paths']['output_dir']}/table11_lambda_sweep.json", sweep_summary)
    for row in sweep_summary:
        print(f"lambda={row['lambda']:.2f} -> macro-F1={row['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
