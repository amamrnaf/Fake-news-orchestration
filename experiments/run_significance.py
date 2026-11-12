"""Pre-registered comparison family (PROTOCOL.md section 7): paired bootstrap
on macro-F1 with Holm correction across the whole family.

Usage: python -m experiments.run_significance --config configs/default.yaml
Requires results/table6_item_predictions.json and results/fusion_item_predictions.json.
"""
from __future__ import annotations

import json
import os

from eval.metrics import macro_f1
from eval.significance import holm_correction, paired_bootstrap_test
from experiments.common import load_config, parse_args
from utils.io import write_json

COMPARISONS = [
    ("Proposed router", "Supervised transformer"),
    ("Domain-rule router", "Supervised transformer"),
    ("LLM + affective retrieval", "LLM + semantic retrieval"),
    ("Performance-weighted fusion", "Supervised transformer"),
    ("Stacked logistic fusion", "Supervised transformer"),
    ("Supervised-transformer-first escalation", "Supervised transformer"),
    ("Performance-weighted fusion", "Confidence ensemble"),
    ("Stacked logistic fusion", "Confidence ensemble"),
    ("Supervised-transformer-first escalation", "Confidence ensemble"),
]
# Router variants 1 and 3 (variant 2 is the proposed router above); included
# once results/router_variants_item_predictions.json exists.
ROUTER_VARIANT_COMPARISONS = [
    ("Original-prompt router", "Supervised transformer"),
    ("Router without evidence-verifier rule or retrieved-example metadata", "Supervised transformer"),
]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]
    out_dir = cfg["paths"]["output_dir"]

    with open(f"{out_dir}/table6_item_predictions.json", encoding="utf-8") as f:
        t6 = json.load(f)
    with open(f"{out_dir}/fusion_item_predictions.json", encoding="utf-8") as f:
        fu = json.load(f)
    if t6["item_ids"] != fu["item_ids"]:
        raise RuntimeError("Table 6 and fusion predictions are not over the same items in the same order")

    gold = t6["gold"]
    labels = {name: p["label"] for name, p in t6["predictions"].items()}
    labels.update({name: p["label"] for name, p in fu["predictions"].items()})

    comparisons = list(COMPARISONS)
    variants_path = f"{out_dir}/router_variants_item_predictions.json"
    if os.path.exists(variants_path):
        with open(variants_path, encoding="utf-8") as f:
            rv = json.load(f)
        if rv["item_ids"] != t6["item_ids"]:
            raise RuntimeError("Router-variant predictions are not over the same items in the same order")
        labels.update({name: p["label"] for name, p in rv["predictions"].items()})
        comparisons += ROUTER_VARIANT_COMPARISONS
    else:
        print(f"note: {variants_path} not found; router variants 1 and 3 left out of the family")

    results = {}
    for a, b in comparisons:
        r = paired_bootstrap_test(gold, labels[a], labels[b], metric_fn=macro_f1, n_samples=cfg["bootstrap_samples"], seed=seed)
        results[f"{a} vs {b}"] = r
    holm = holm_correction({k: r.p_value for k, r in results.items()}, alpha=cfg["evaluation"]["holm_alpha"])

    rows = [
        {
            "comparison": k,
            "macro_f1_difference": r.observed_diff,
            "ci_95_low": r.ci_low,
            "ci_95_high": r.ci_high,
            "p_value": r.p_value,
            "holm_adjusted_p": holm[k][0],
            "significant_after_holm": holm[k][1],
        }
        for k, r in results.items()
    ]
    write_json(f"{out_dir}/significance_family.json", rows)
    for row in rows:
        flag = "significant" if row["significant_after_holm"] else "not significant"
        print(f"{row['comparison']:70s} | diff={row['macro_f1_difference']:+.3f} "
              f"[{row['ci_95_low']:+.3f}, {row['ci_95_high']:+.3f}] | Holm p={row['holm_adjusted_p']:.4f} ({flag})")


if __name__ == "__main__":
    main()
