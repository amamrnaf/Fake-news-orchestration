"""Produces Table 6: main AMTCele veracity results, comparing individual
detectors/retrieval-mode baselines, rule-based/ensemble baselines, the
proposed router, and the oracle. Also runs the RQ1 significance test
(proposed router vs. strongest individual detector) via paired bootstrap +
Holm correction (Sec 4.5).

Usage: python -m experiments.run_main_veracity --config configs/default.yaml
"""
from __future__ import annotations

from domain.predict import predict_domain_batch
from eval.efficiency import aggregate_efficiency, record_from_outputs
from eval.metrics import macro_f1, veracity_report
from eval.significance import holm_correction, paired_bootstrap_test
from experiments.common import build_catalogue_for, build_encoders, build_registry, build_routing_domain_classifier, load_and_split_amtcele, load_config, parse_args, split_by
from retrieval.index import build_index_for_pool
from routing.baselines import AllToolLLMAdvisor, DomainRuleRouter, confidence_weighted_ensemble, majority_class_predictions
from routing.llm_router import LLMRouter
from routing.oracle import oracle_select, outputs_from_decisions
from detectors.t2_retrieval_llm import RetrievedExampleLLMDetector
from utils.io import write_json

_INDIVIDUAL_DETECTOR_METHODS = (
    "Supervised transformer",
    "LLM zero-shot",
    "LLM + random examples",
    "LLM + semantic retrieval",
    "LLM + affective retrieval",
)


def _report_row(name: str, gold, outputs, tools_per_input: float) -> dict:
    y_pred = [o.label for o in outputs]
    y_prob = [o.prob_fake for o in outputs]
    report = veracity_report(gold, y_pred, y_prob)
    return {"method": name, **report, "tools_per_input": tools_per_input, "_outputs": outputs}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    train_items, val_items, test_items = split_by(items, "train"), split_by(items, "val"), split_by(items, "test")
    gold = [it.label for it in test_items]

    registry = build_registry(cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed, t1_checkpoint_key=f"seed{seed}")
    catalogue, val_outputs_by_tool = build_catalogue_for(registry, val_items)

    outputs_by_tool = {tid: registry.get(tid).predict_batch(test_items) for tid in registry.all_ids()}

    # Domain predictions for router-based methods.
    domain_model = build_routing_domain_classifier(cfg, train_items, val_items, seed, checkpoint_key=f"seed{seed}")
    domain_preds = predict_domain_batch(domain_model, [it.text for it in test_items])

    retrieval_index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)
    test_sem = semantic_encoder.encode([it.text for it in test_items])
    test_aff = affective_encoder.encode([it.text for it in test_items])
    retrieved = retrieval_index.search_batch(test_sem, test_aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"])

    rows = []
    rows.append(_report_row("Majority class", gold, majority_class_predictions(train_items, len(test_items)), 0.0))
    rows.append(_report_row("Supervised transformer", gold, outputs_by_tool["T1"], 1.0))

    # Table 4/Table 6 LLM retrieval-mode baselines: same LLM backend as T2, differing
    # only in example-selection strategy. Distinct from the tuned affective_semantic
    # T2 detector in `registry`, which the proposed router and ensembles use.
    retrieval_mode_baselines = [
        ("LLM zero-shot", "zero_shot"),
        ("LLM + random examples", "random"),
        ("LLM + semantic retrieval", "semantic"),
        ("LLM + affective retrieval", "affective"),
    ]
    for row_name, mode in retrieval_mode_baselines:
        detector = RetrievedExampleLLMDetector(
            llm=registry.get("T2").llm,
            index=retrieval_index,
            semantic_encoder=semantic_encoder,
            affective_encoder=affective_encoder,
            mode=mode,
            k=cfg["retrieval"]["k"],
            alpha=cfg["retrieval"]["alpha"],
            max_new_tokens=cfg["llm"]["max_new_tokens"],
            seed=seed,
        )
        rows.append(_report_row(row_name, gold, detector.predict_batch(test_items), 1.0))

    ensemble_outputs = [
        confidence_weighted_ensemble({tid: outputs_by_tool[tid][i] for tid in registry.all_ids()})
        for i in range(len(test_items))
    ]
    rows.append(_report_row("Confidence ensemble", gold, ensemble_outputs, 3.0))

    advisor = AllToolLLMAdvisor(llm=registry.get("T2").llm)
    advisor_outputs = [
        advisor.synthesise(test_items[i], {tid: outputs_by_tool[tid][i] for tid in registry.all_ids()})
        for i in range(len(test_items))
    ]
    rows.append(_report_row("All-tool LLM advisor", gold, advisor_outputs, 3.0))

    domain_rule = DomainRuleRouter(catalogue)
    dr_decisions = [domain_rule.select(test_items[i], domain_preds[i]) for i in range(len(test_items))]
    dr_outputs = outputs_from_decisions(dr_decisions, outputs_by_tool)
    rows.append(_report_row("Domain-rule router", gold, dr_outputs, 1.0))

    router = LLMRouter(
        llm=registry.get("T2").llm,
        tool_descriptions=registry.catalogue_descriptions(),
        validation_catalogue=catalogue,
        mode=cfg["router"]["mode"],
        max_retries=cfg["router"]["max_retries"],
        max_new_tokens=cfg["router"]["max_new_tokens"],
    )
    affective_summary_all = affective_encoder.encode([it.text for it in test_items])
    proposed_decisions = [
        router.select(
            test_items[i],
            domain_preds[i],
            affective_summary={affective_encoder.labels[j]: float(affective_summary_all[i][j]) for j in range(len(affective_encoder.labels))},
            retrieved_examples=retrieved[i],
        )
        for i in range(len(test_items))
    ]
    proposed_outputs = outputs_from_decisions(proposed_decisions, outputs_by_tool)
    proposed_tools_per_input = sum(1 + d.llm_calls for d in proposed_decisions) / len(proposed_decisions)
    rows.append(_report_row("Proposed router", gold, proposed_outputs, proposed_tools_per_input))

    oracle_decisions = oracle_select(test_items, outputs_by_tool)
    oracle_outputs = outputs_from_decisions(oracle_decisions, outputs_by_tool)
    rows.append(_report_row("Oracle router", gold, oracle_outputs, 1.0))

    # RQ1: significance test, proposed router vs. strongest individual detector.
    individual_rows = [r for r in rows if r["method"] in _INDIVIDUAL_DETECTOR_METHODS]
    strongest = max(individual_rows, key=lambda r: r["macro_f1"])
    bootstrap = paired_bootstrap_test(
        gold,
        [o.label for o in proposed_outputs],
        [o.label for o in strongest["_outputs"]],
        metric_fn=macro_f1,
        n_samples=cfg["bootstrap_samples"],
        seed=seed,
    )
    holm = holm_correction({"proposed_vs_strongest_individual": bootstrap.p_value}, alpha=cfg["evaluation"]["holm_alpha"])

    for row in rows:
        row.pop("_outputs")
    write_json(f"{cfg['paths']['output_dir']}/table6_main_veracity.json", rows)
    write_json(
        f"{cfg['paths']['output_dir']}/rq1_significance.json",
        {"strongest_individual": strongest["method"], "bootstrap": bootstrap.__dict__, "holm": {k: v for k, v in holm.items()}},
    )

    for row in rows:
        print(f"{row['method']:42s} | Acc={row['accuracy']:.3f} | Macro-F1={row['macro_f1']:.3f} | "
              f"ECE={row['ece']:.3f} | Tools/input={row['tools_per_input']:.2f}")
    print(f"\nRQ1 bootstrap diff={bootstrap.observed_diff:.4f}, p={bootstrap.p_value:.4f} "
          f"(Holm-adjusted p={holm['proposed_vs_strongest_individual'][0]:.4f})")


if __name__ == "__main__":
    main()
