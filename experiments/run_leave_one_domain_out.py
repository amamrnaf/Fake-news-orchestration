"""Produces Table 7: macro-F1 under leave-one-domain-out evaluation (Sec 4.2).

For each held-out domain, detectors are trained/retrieval-indexed only on the
remaining source domains; no target-domain validation score is available to
the router, so `DomainRuleRouter`/`LLMRouter` must generalise from source
domains via the predicted-domain distribution and retrieval similarity alone.

Usage: python -m experiments.run_leave_one_domain_out --config configs/default.yaml
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

from data.amtcele import load_amtcele
from data.splits import available_groups, leave_one_group_out, stratified_split
from domain.predict import predict_domain_batch
from domain.train import TrainConfig, train_one_config
from eval.metrics import macro_f1
from experiments.common import build_catalogue_for, build_encoders, build_registry, load_config, parse_args
from retrieval.index import build_index_for_pool
from routing.baselines import DomainRuleRouter
from routing.llm_router import LLMRouter
from routing.oracle import oracle_select, outputs_from_decisions
from utils.cache import cached_map, load_jsonl_cache
from utils.io import write_json
from utils.lm_studio import load_llm, unload_llm
from utils.types import DOMAINS, DomainPrediction, RouterDecision


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    all_items = load_amtcele(cfg["paths"]["amtcele_raw"])
    domains_present = [d for d in DOMAINS if d in available_groups(all_items)]

    per_domain_rows: List[dict] = []
    for fold_i, held_out_domain in enumerate(domains_present):
        print(f"\n=== LODO fold {fold_i + 1}/{len(domains_present)}: held_out_domain={held_out_domain!r} ===", flush=True)
        split_dict = leave_one_group_out(all_items, held_out_domain)

        # Source domains only are split into train/val; the held-out domain
        # itself is evaluated wholesale (Sec 4.2: no target-domain validation
        # score is available to the router).
        source_train_val = stratified_split(
            split_dict["source"], train_size=0.85, val_size=0.15, test_size=0.0, seed=seed, semantic_encoder=semantic_encoder
        )
        train_items = [it for it in source_train_val if it.split == "train"]
        val_items = [it for it in source_train_val if it.split in ("val", "test")]
        held_out_items = split_dict["held_out"]
        gold = [it.label for it in held_out_items]

        # Unload the LLM for the whole GPU-training window (T1 inside
        # build_registry, then the domain classifier) so both get full VRAM
        # headroom instead of sharing ~3GB with LM Studio's resident model.
        # Confirmed empirically: unload is near-instant, reload ~5s -- cheap
        # against this fold's LLM-call cost (T2/T3/router), so doing this
        # once per fold (not per training call) keeps overhead negligible.
        # Cache per fold, not shared across folds: T2's retrieval index (and
        # therefore its predictions) genuinely differs fold to fold, so a
        # cached item_id prediction from fold 1 would be wrong reused in
        # fold 2. Safe within a fold, though -- a resumed run of the same
        # fold has the same retrieval index.
        fold_cache_dir = os.path.join(cfg["paths"]["output_dir"], "prediction_cache", f"lodo_{held_out_domain}")

        # Resume shortcut: a fold that was already trained and whose outputs
        # are all cached does not need its models trained again. T1 is
        # skipped only when every validation and held-out T1 prediction is
        # cached; the domain classifier only when its held-out predictions
        # were saved by an earlier run of this fold.
        t1_val_cached = load_jsonl_cache(os.path.join(fold_cache_dir, "T1_val_cache.jsonl"))
        t1_heldout_cached = load_jsonl_cache(os.path.join(fold_cache_dir, "T1_heldout_cache.jsonl"))
        skip_t1 = all(it.id in t1_val_cached for it in val_items) and all(
            it.id in t1_heldout_cached for it in held_out_items
        )
        domain_preds_path = os.path.join(fold_cache_dir, "domain_preds_heldout.json")
        domain_preds = None
        if os.path.exists(domain_preds_path):
            with open(domain_preds_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            if [r["item_id"] for r in saved] == [it.id for it in held_out_items]:
                domain_preds = [DomainPrediction(**r["prediction"]) for r in saved]
                print(f"[domain] loaded saved held-out predictions from {domain_preds_path}", flush=True)

        need_training = not skip_t1 or domain_preds is None
        if need_training:
            unload_llm(cfg["llm"]["model_name"])
        registry = build_registry(
            cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed,
            skip_t1_training=skip_t1,
        )

        if domain_preds is None:
            dc_cfg = cfg["domain_classifier"]
            dom_cfg = TrainConfig(
                model_name=dc_cfg["candidate_encoders"][0], lr=dc_cfg["lr_grid"][0],
                batch_size=dc_cfg["batch_size_grid"][0], max_length=dc_cfg["max_length_grid"][0], seed=seed,
            )
            domain_model, _ = train_one_config(train_items, val_items, dom_cfg)
            domain_preds = predict_domain_batch(domain_model, [it.text for it in held_out_items])
            del domain_model
            os.makedirs(fold_cache_dir, exist_ok=True)
            with open(domain_preds_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [{"item_id": it.id, "prediction": asdict(p)} for it, p in zip(held_out_items, domain_preds)], fh
                )
        if need_training:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # hand training memory back before the LLM reloads
            load_llm(cfg["llm"]["model_name"])

        catalogue, _ = build_catalogue_for(registry, val_items, cache_dir=fold_cache_dir)
        outputs_by_tool = {
            tid: registry.get(tid).predict_batch(
                held_out_items, cache_path=os.path.join(fold_cache_dir, f"{tid}_heldout_cache.jsonl")
            )
            for tid in registry.all_ids()
        }

        best_single_f1 = max(macro_f1(gold, [o.label for o in outputs_by_tool[tid]]) for tid in registry.all_ids())

        domain_rule = DomainRuleRouter(catalogue)
        dr_decisions = [domain_rule.select(held_out_items[i], domain_preds[i]) for i in range(len(held_out_items))]
        dr_f1 = macro_f1(gold, [o.label for o in outputs_from_decisions(dr_decisions, outputs_by_tool)])

        index = build_index_for_pool(train_items + val_items, semantic_encoder, affective_encoder)
        sem = semantic_encoder.encode([it.text for it in held_out_items])
        aff = affective_encoder.encode([it.text for it in held_out_items])
        retrieved = index.search_batch(sem, aff, k=cfg["retrieval"]["k"], alpha=cfg["retrieval"]["alpha"])

        router = LLMRouter(
            llm=registry.get("T2").llm, tool_descriptions=registry.catalogue_descriptions(),
            validation_catalogue=catalogue, mode=cfg["router"]["mode"],
            max_retries=cfg["router"]["max_retries"], max_new_tokens=cfg["router"]["max_new_tokens"],
        )
        aff_summary = affective_encoder.encode([it.text for it in held_out_items])

        def _select(i: int) -> RouterDecision:
            return router.select(
                held_out_items[i], domain_preds[i],
                affective_summary={affective_encoder.labels[j]: float(aff_summary[i][j]) for j in range(len(affective_encoder.labels))},
                retrieved_examples=retrieved[i],
            )

        proposed_decisions = cached_map(
            [it.id for it in held_out_items],
            _select,
            lambda d: RouterDecision(**d),
            cache_path=os.path.join(fold_cache_dir, "router_heldout_cache.jsonl"),
            label=f"router:{held_out_domain}",
        )
        proposed_f1 = macro_f1(gold, [o.label for o in outputs_from_decisions(proposed_decisions, outputs_by_tool)])

        oracle_decisions = oracle_select(held_out_items, outputs_by_tool)
        oracle_f1 = macro_f1(gold, [o.label for o in outputs_from_decisions(oracle_decisions, outputs_by_tool)])

        per_domain_rows.append(
            {
                "held_out_domain": held_out_domain,
                "best_single_tool": best_single_f1,
                "domain_rule_router": dr_f1,
                "proposed_router": proposed_f1,
                "oracle": oracle_f1,
            }
        )

    macro_avg = {
        key: sum(r[key] for r in per_domain_rows) / len(per_domain_rows)
        for key in ("best_single_tool", "domain_rule_router", "proposed_router", "oracle")
    }
    per_domain_rows.append({"held_out_domain": "Macro-average", **macro_avg})

    write_json(f"{cfg['paths']['output_dir']}/table7_leave_one_domain_out.json", per_domain_rows)
    for row in per_domain_rows:
        print(
            f"{row['held_out_domain']:15s} | best={row['best_single_tool']:.3f} | "
            f"domain-rule={row['domain_rule_router']:.3f} | proposed={row['proposed_router']:.3f} | "
            f"oracle={row['oracle']:.3f}"
        )


if __name__ == "__main__":
    main()
