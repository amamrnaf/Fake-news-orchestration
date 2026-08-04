"""Produces Table 2: validation-only detector catalogue (Sec 3.5).

Usage: python -m experiments.run_catalogue --config configs/default.yaml
"""
from __future__ import annotations

from perf_catalogue.build_catalogue import catalogue_to_table2_rows, save_catalogue
from experiments.common import build_catalogue_for, build_encoders, build_registry, load_and_split_amtcele, load_config, parse_args, prediction_cache_dir, split_by


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, affective_encoder = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    train_items, val_items = split_by(items, "train"), split_by(items, "val")

    registry = build_registry(cfg, train_items, val_items, train_items + val_items, semantic_encoder, affective_encoder, seed, t1_checkpoint_key=f"seed{seed}")
    cache_dir = prediction_cache_dir(cfg, seed)
    catalogue, _ = build_catalogue_for(registry, val_items, cache_dir=cache_dir)

    save_catalogue(catalogue, f"{cfg['paths']['output_dir']}/table2_validation_catalogue.json")
    rows = catalogue_to_table2_rows(catalogue)
    for row in rows:
        print(
            f"{row['domain']:15s} | T1={row['T1_F1']} | T2={row['T2_F1']} | T3={row['T3_F1']} | "
            f"best={row['best_validation_tool']}"
        )


if __name__ == "__main__":
    main()
