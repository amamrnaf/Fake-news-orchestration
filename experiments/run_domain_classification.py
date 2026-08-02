"""Produces Table 5: domain-classification performance on the AMTCele test
partition, for the three input variants of Sec 3.3 (text only / + semantic
neighbours / + affective-semantic neighbours), selecting the principal
classifier per candidate encoder by validation macro-F1.

Usage: python -m experiments.run_domain_classification --config configs/default.yaml
"""
from __future__ import annotations

from experiments.common import build_encoders, load_and_split_amtcele, load_config, parse_args, split_by
from domain.predict import predict_domain_batch
from domain.train import grid_search_domain_classifier
from eval.metrics import domain_report
from retrieval.index import build_index_for_pool
from utils.io import write_json
from utils.types import DOMAINS


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    train_items, val_items, test_items = split_by(items, "train"), split_by(items, "val"), split_by(items, "test")

    rows = []
    encoders = cfg["domain_classifier"]["candidate_encoders"]

    # Variant (i): text only, per candidate encoder.
    per_encoder_best = {}
    for encoder_name in encoders:
        model, train_cfg, val_f1 = grid_search_domain_classifier(
            [encoder_name],
            train_items,
            val_items,
            lr_grid=cfg["domain_classifier"]["lr_grid"],
            batch_size_grid=cfg["domain_classifier"]["batch_size_grid"],
            max_len_grid=cfg["domain_classifier"]["max_length_grid"],
            seed=seed,
        )
        per_encoder_best[encoder_name] = (model, train_cfg, val_f1)
        preds = predict_domain_batch(model, [it.text for it in test_items])
        report = domain_report(
            [it.domain for it in test_items],
            [p.predicted_domain for p in preds],
            [p.distribution for p in preds],
            [p.confidence for p in preds],
            DOMAINS,
        )
        rows.append({"classifier": encoder_name, "input": "Text only", **report, "val_macro_f1": val_f1})

    # Best encoder by validation macro-F1, reused for variants (ii) and (iii).
    best_encoder_name = max(per_encoder_best, key=lambda k: per_encoder_best[k][2])
    best_model, best_cfg, _ = per_encoder_best[best_encoder_name]

    retrieval_pool = train_items + val_items
    index = build_index_for_pool(retrieval_pool, semantic_encoder, affective_encoder)

    for variant_name, alpha in (("Text + semantic neighbours", 1.0), ("Text + affective-semantic neighbours", cfg["retrieval"]["alpha"])):
        test_sem = semantic_encoder.encode([it.text for it in test_items])
        test_aff = affective_encoder.encode([it.text for it in test_items])
        neighbours = index.search_batch(test_sem, test_aff, k=cfg["retrieval"]["k"], alpha=alpha)
        preds = predict_domain_batch(best_model, [it.text for it in test_items], neighbours=neighbours)
        report = domain_report(
            [it.domain for it in test_items],
            [p.predicted_domain for p in preds],
            [p.distribution for p in preds],
            [p.confidence for p in preds],
            DOMAINS,
        )
        rows.append({"classifier": f"Best encoder ({best_encoder_name})", "input": variant_name, **report})

    write_json(f"{cfg['paths']['output_dir']}/table5_domain_classification.json", rows)
    for row in rows:
        print(f"{row['classifier']:35s} | {row['input']:38s} | macro-F1={row['macro_f1']:.3f} | "
              f"bal.acc={row['balanced_accuracy']:.3f} | ECE={row['ece']:.3f}")


if __name__ == "__main__":
    main()
