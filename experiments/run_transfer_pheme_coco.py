"""Produces Table 9: secondary event- and topic-transfer results on PHEME
(leave-one-event-out) and COCO (held-out topic groups) — Sec 4.1, Table 3.

The detector registry (T1/T2/T3) and domain classifier are trained once on
AMTCele's permitted source-domain examples, matching Sec 4.1's framing of
PHEME/COCO as secondary *transfer* tests rather than in-domain benchmarks:
no PHEME/COCO label is used for detector fitting. `leave_one_group_out` is
applied per event/topic purely as the evaluation protocol (Sec 4.2), so the
loop below evaluates each held-out group with the same fixed, AMTCele-trained
tool bank and pools predictions across groups for the aggregate row.

Usage: python -m experiments.run_transfer_pheme_coco --config configs/default.yaml
"""
from __future__ import annotations

from data.amtcele import load_amtcele
from data.coco import load_coco
from data.pheme import load_pheme
from data.splits import available_groups, leave_one_group_out, stratified_split
from domain.predict import predict_domain_batch
from domain.train import TrainConfig, train_one_config
from eval.metrics import macro_f1
from experiments.common import build_catalogue_for, build_encoders, build_registry, load_config, parse_args
from retrieval.index import build_index_for_pool
from routing.baselines import confidence_weighted_ensemble
from routing.llm_router import LLMRouter
from routing.oracle import outputs_from_decisions
from utils.io import write_json
from utils.types import NewsItem, TOOL_IDS


def _evaluate_transfer_dataset(dataset_name, items, cfg, registry, catalogue, domain_model, index, semantic_encoder, affective_encoder, seed):
    groups = available_groups(items)
    all_gold, all_t1, all_t2, all_ensemble, all_proposed = [], [], [], [], []

    for group in groups:
        held_out_items = leave_one_group_out(items, group)["held_out"]
        if not held_out_items:
            continue
        gold = [it.label for it in held_out_items]
        outputs_by_tool = {tid: registry.get(tid).predict_batch(held_out_items) for tid in registry.all_ids()}

        domain_preds = predict_domain_batch(domain_model, [it.text for it in held_out_items])
        sem = semantic_encoder.encode([it.text for it in held_out_items])
        aff = affective_encoder.encode([it.text for it in held_out_items])
        retrieved = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=0.0)  # "Affective LLM" column -> affective-only retrieval

        router = LLMRouter(
            llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
            validation_catalogue=catalogue, mode=cfg["router"]["mode"],
            max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
        )
        aff_summary = affective_encoder.encode([it.text for it in held_out_items])
        decisions = [
            router.select(
                held_out_items[i], domain_preds[i],
                affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
                retrieved_examples=retrieved[i],
            )
            for i in range(len(held_out_items))
        ]
        proposed_outputs = outputs_from_decisions(decisions, outputs_by_tool)
        ensemble_outputs = [
            confidence_weighted_ensemble({tid: outputs_by_tool[tid][i] for tid in registry.all_ids()})
            for i in range(len(held_out_items))
        ]

        all_gold.extend(gold)
        all_t1.extend(o.label for o in outputs_by_tool["T1"])
        all_t2.extend(o.label for o in outputs_by_tool["T2"])
        all_ensemble.extend(o.label for o in ensemble_outputs)
        all_proposed.extend(o.label for o in proposed_outputs)

    return {
        "dataset_split": dataset_name,
        "supervised_tool_macro_f1": macro_f1(all_gold, all_t1) if all_gold else None,
        "affective_llm_macro_f1": macro_f1(all_gold, all_t2) if all_gold else None,
        "fixed_ensemble_macro_f1": macro_f1(all_gold, all_ensemble) if all_gold else None,
        "proposed_router_macro_f1": macro_f1(all_gold, all_proposed) if all_gold else None,
        "n_items": len(all_gold),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    amtcele_items = stratified_split(load_amtcele(cfg["paths"]["amtcele_raw"]), seed=seed, semantic_encoder=semantic_encoder)
    train_items = [it for it in amtcele_items if it.split == "train"]
    val_items = [it for it in amtcele_items if it.split == "val"]

    registry = build_registry(cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed)
    catalogue, _ = build_catalogue_for(registry, val_items)

    dc_cfg = cfg["domain_classifier"]
    dom_cfg = TrainConfig(
        model_name=dc_cfg["candidate_encoders"][0], lr=dc_cfg["lr_grid"][0],
        batch_size=dc_cfg["batch_size_grid"][0], max_length=dc_cfg["max_length_grid"][0], seed=seed,
    )
    domain_model, _ = train_one_config(train_items, val_items, dom_cfg)
    index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)

    rows = []
    try:
        pheme_items = load_pheme(cfg["paths"]["pheme_raw"])
        rows.append(_evaluate_transfer_dataset(
            "PHEME leave-one-event-out", pheme_items, cfg, registry, catalogue, domain_model, index, semantic_encoder, affective_encoder, seed
        ))
    except FileNotFoundError as exc:
        print(f"[skip] PHEME: {exc}")

    try:
        coco_items = load_coco(cfg["paths"]["coco_csv"])
        rows.append(_evaluate_transfer_dataset(
            "COCO held-out topics", coco_items, cfg, registry, catalogue, domain_model, index, semantic_encoder, affective_encoder, seed
        ))
    except FileNotFoundError as exc:
        print(f"[skip] COCO: {exc}")

    write_json(f"{cfg['paths']['output_dir']}/table9_pheme_coco_transfer.json", rows)
    for row in rows:
        print(
            f"{row['dataset_split']:28s} | T1={row['supervised_tool_macro_f1']} | "
            f"AffLLM={row['affective_llm_macro_f1']} | Ensemble={row['fixed_ensemble_macro_f1']} | "
            f"Proposed={row['proposed_router_macro_f1']} | n={row['n_items']}"
        )


if __name__ == "__main__":
    main()
