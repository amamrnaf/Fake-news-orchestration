"""Validation-only performance catalogue (Sec 3.5, Table 2).

'All candidate detectors are run on the same validation partition. For each
domain, the catalogue stores macro-F1, class recall, calibration error,
latency, and failure frequency... a compact description of each detector's
observed strengths.'

The catalogue is built exclusively from validation predictions (never test)
to avoid circular evaluation (Sec 3, 'To avoid circular evaluation, the
detector catalogue is constructed exclusively from training and validation
predictions, while the held-out test set is used only for final reporting.')
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from detectors.registry import DetectorRegistry
from eval.metrics import expected_calibration_error, macro_f1, per_class_precision_recall
from utils.io import write_json
from utils.types import DetectorOutput, NewsItem


def _group_by_domain(items: Sequence[NewsItem]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(items):
        key = item.domain or "__all__"
        groups[key].append(i)
    return groups


def run_detectors_on_validation(
    registry: DetectorRegistry, val_items: Sequence[NewsItem]
) -> Dict[str, List[DetectorOutput]]:
    """Runs every tool on the same validation partition (common protocol,
    not incomparable numbers copied from prior papers, Sec 3.4)."""
    return {tool_id: registry.get(tool_id).predict_batch(list(val_items)) for tool_id in registry.all_ids()}


def build_validation_catalogue(
    val_items: Sequence[NewsItem], outputs_by_tool: Dict[str, List[DetectorOutput]]
) -> Dict[str, Dict[str, dict]]:
    """Returns {domain: {tool_id: {macro_f1, class_recall, ece, mean_latency_s,
    failure_rate}}} plus a 'best validation tool' per domain (Table 2)."""
    domain_groups = _group_by_domain(val_items)
    gold = [it.label for it in val_items]

    catalogue: Dict[str, Dict[str, dict]] = {}
    for domain, idxs in domain_groups.items():
        domain_entry: Dict[str, dict] = {}
        best_tool, best_f1 = None, -1.0
        for tool_id, outputs in outputs_by_tool.items():
            y_true = [gold[i] for i in idxs]
            y_pred = [outputs[i].label for i in idxs]
            y_conf = [max(outputs[i].prob_fake, 1 - outputs[i].prob_fake) for i in idxs]
            failures = [outputs[i].failed for i in idxs]
            latencies = [outputs[i].latency_s for i in idxs]

            f1 = macro_f1(y_true, y_pred)
            entry = {
                "macro_f1": f1,
                "class_recall": per_class_precision_recall(y_true, y_pred),
                "ece": expected_calibration_error(y_true, y_pred, y_conf),
                "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
                "failure_rate": float(np.mean(failures)) if failures else 0.0,
                "n_items": len(idxs),
            }
            domain_entry[tool_id] = entry
            if f1 > best_f1:
                best_tool, best_f1 = tool_id, f1
        domain_entry["best_validation_tool"] = best_tool
        catalogue[domain] = domain_entry
    return catalogue


def catalogue_to_table2_rows(catalogue: Dict[str, Dict[str, dict]]) -> List[dict]:
    """Flattens the catalogue into rows matching Table 2's columns
    (Domain, T1 F1, T2 F1, T3 F1, Best validation tool)."""
    rows = []
    for domain, entry in catalogue.items():
        if domain == "__all__":
            continue
        rows.append(
            {
                "domain": domain,
                "T1_F1": entry.get("T1", {}).get("macro_f1"),
                "T2_F1": entry.get("T2", {}).get("macro_f1"),
                "T3_F1": entry.get("T3", {}).get("macro_f1"),
                "best_validation_tool": entry.get("best_validation_tool"),
            }
        )
    return rows


def save_catalogue(catalogue: Dict[str, Dict[str, dict]], path: str) -> None:
    write_json(path, catalogue)
