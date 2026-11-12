"""Soft fusion and supervised-transformer-first escalation (PROTOCOL.md, router
variants 4 and 5). Uses only cached detector predictions -- no LLM calls -- and
chooses every setting (weights, regularisation, threshold) on the validation
partition; the test partition is used once, for reporting.

Variants:
  Performance-weighted fusion  - average of the three detectors' P(fake),
                                 weighted by each detector's validation macro-F1.
  Stacked logistic fusion      - logistic regression on the three P(fake)
                                 values, fitted on validation (regularisation by
                                 5-fold cross-validation within validation).
  Escalation                   - supervised transformer's answer when its
                                 confidence >= threshold, else performance-
                                 weighted fusion; threshold chosen on validation.

Usage: python -m experiments.run_fusion --config configs/default.yaml
Requires the seed's T1/T2/T3 val and test caches (run_catalogue + run_main_veracity).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from eval.metrics import macro_f1, veracity_report
from experiments.common import build_encoders, cache_path_for, load_and_split_amtcele, load_config, parse_args, prediction_cache_dir, split_by
from routing.baselines import confidence_weighted_ensemble
from utils.cache import load_jsonl_cache
from utils.io import write_json
from utils.types import DetectorOutput, NewsItem

TOOLS = ("T1", "T2", "T3")


def _load_cached(path: str, items: List[NewsItem]) -> List[DetectorOutput]:
    by_id = load_jsonl_cache(path)
    missing = [it.id for it in items if it.id not in by_id]
    if missing:
        raise RuntimeError(f"{path} is missing {len(missing)} items -- run the detectors first")
    return [DetectorOutput(**by_id[it.id]) for it in items]


def _probs(outputs: Dict[str, List[DetectorOutput]]) -> np.ndarray:
    # Failed calls carry prob_fake=0.5 already, i.e. an abstention.
    return np.array([[o.prob_fake for o in outputs[t]] for t in TOOLS]).T


def _weighted(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return probs @ weights / weights.sum()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]

    semantic_encoder, _ = build_encoders(cfg)
    items = load_and_split_amtcele(cfg, seed, semantic_encoder=semantic_encoder)
    val_items, test_items = split_by(items, "val"), split_by(items, "test")
    cache_dir = prediction_cache_dir(cfg, seed)

    val_out = {t: _load_cached(cache_path_for(cache_dir, t, "val"), val_items) for t in TOOLS}
    test_out = {t: _load_cached(cache_path_for(cache_dir, t, "test"), test_items) for t in TOOLS}
    y_val = np.array([it.label for it in val_items])
    y_test = np.array([it.label for it in test_items])
    p_val, p_test = _probs(val_out), _probs(test_out)

    # 1. Performance-weighted fusion.
    weights = np.array([macro_f1(y_val, [o.label for o in val_out[t]]) for t in TOOLS])
    pw_test = _weighted(p_test, weights)

    # 2. Stacked logistic fusion.
    stacker = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", random_state=seed, max_iter=1000)
    stacker.fit(p_val, y_val)
    st_test = stacker.predict_proba(p_test)[:, 1]

    # 3. Escalation: threshold on the supervised transformer's confidence.
    def escalate(p: np.ndarray, threshold: float):
        t1_conf = np.maximum(p[:, 0], 1 - p[:, 0])
        escalated = t1_conf < threshold
        return np.where(escalated, _weighted(p, weights), p[:, 0]), escalated

    grid = np.round(np.arange(0.50, 1.0001, 0.01), 2)
    # Ties go to the lowest threshold, i.e. the fewest (costly) escalations.
    best_threshold, best_val_f1 = grid[0], -1.0
    for th in grid:
        prob, _ = escalate(p_val, th)
        f1 = macro_f1(y_val, (prob >= 0.5).astype(int))
        if f1 > best_val_f1 + 1e-12:
            best_threshold, best_val_f1 = float(th), f1
    esc_test, esc_mask = escalate(p_test, best_threshold)

    ensemble_test = np.array([
        confidence_weighted_ensemble({t: test_out[t][i] for t in TOOLS}).prob_fake for i in range(len(test_items))
    ])

    systems = {
        "Supervised transformer": p_test[:, 0],
        "Confidence ensemble": ensemble_test,
        "Performance-weighted fusion": pw_test,
        "Stacked logistic fusion": st_test,
        "Supervised-transformer-first escalation": esc_test,
    }
    rows = []
    for name, prob in systems.items():
        pred = (prob >= 0.5).astype(int)
        rows.append({"method": name, **veracity_report(y_test.tolist(), pred.tolist(), prob.tolist())})

    out_dir = cfg["paths"]["output_dir"]
    write_json(f"{out_dir}/fusion_results.json", {
        "rows": rows,
        "selected_on_validation": {
            "fusion_weights_val_macro_f1": dict(zip(TOOLS, weights.tolist())),
            "stacker_C": float(stacker.C_[0]),
            "stacker_coefficients": dict(zip(TOOLS, stacker.coef_[0].tolist())),
            "escalation_threshold": best_threshold,
            "escalation_val_macro_f1": best_val_f1,
            "escalation_rate_test": float(esc_mask.mean()),
        },
    })
    write_json(f"{out_dir}/fusion_item_predictions.json", {
        "item_ids": [it.id for it in test_items],
        "gold": y_test.tolist(),
        "predictions": {
            name: {"label": (prob >= 0.5).astype(int).tolist(), "prob_fake": prob.tolist()} for name, prob in systems.items()
        },
        "escalated": esc_mask.tolist(),
    })

    for r in rows:
        print(f"{r['method']:42s} | Acc={r['accuracy']:.3f} | Macro-F1={r['macro_f1']:.3f} | ECE={r['ece']:.3f}")
    print(f"weights={dict(zip(TOOLS, np.round(weights, 3)))} | stacker C={stacker.C_[0]:.4g} "
          f"| escalation threshold={best_threshold} (val macro-F1={best_val_f1:.3f}, test escalation rate={esc_mask.mean():.2%})")


if __name__ == "__main__":
    main()
