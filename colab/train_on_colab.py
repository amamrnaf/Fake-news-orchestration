"""Source for colab/train_on_colab.ipynb, cell content only (see that file's
markdown cells for instructions). Kept as a plain .py alongside the notebook
so the training logic is diffable/reviewable outside Jupyter's JSON wrapper;
edit this if you need to change what gets trained, then regenerate the
notebook cells to match.

What this does, end to end:
  1. Loads AMTCele and reproduces the exact same train/val/test split the
     local scripts use (same seed, same near-duplicate-aware stratified
     split) -- this is what makes a Colab-trained checkpoint valid to drop
     into the local pipeline instead of a coincidence.
  2. Fine-tunes T1 (RoBERTa-base by default, per configs/default.yaml) for
     every seed in cfg["seeds"], saving each to checkpoints/t1/seed<N>/.
     Reused by run_catalogue / run_main_veracity / run_ablations /
     run_routing_analysis / run_efficiency (NOT run_leave_one_domain_out,
     which uses a different split per fold).
  3. Trains the lightweight single-config domain classifier used purely as
     routing context in run_main_veracity/run_ablations/run_routing_analysis/
     run_efficiency (NOT the Table 5 grid search below -- that one's result
     is never reused elsewhere), once per seed, saving to
     checkpoints/quick_domain_classifier/seed<N>/.
  4. Runs the full domain-classifier grid search (Table 5: 3 encoders x 12
     hyperparameter combos each) for cfg["seeds"][0], saving the winning
     config per encoder to checkpoints/domain_classifier/<encoder>/.
  5. Zips checkpoints/ for download.
"""
from __future__ import annotations

import yaml

from data.amtcele import load_amtcele
from data.splits import stratified_split
from detectors.t1_supervised import SupervisedTransformerDetector, T1Config
from domain.model import DomainClassifierModel
from domain.train import TrainConfig, grid_search_domain_classifier, train_one_config
from retrieval.embeddings import SemanticEncoder


def load_and_split(cfg: dict, seed: int, semantic_encoder):
    items = load_amtcele(cfg["paths"]["amtcele_raw"])
    return stratified_split(
        items,
        train_size=cfg["splits"]["train_size"],
        val_size=cfg["splits"]["val_size"],
        test_size=cfg["splits"]["test_size"],
        seed=seed,
        stratify_by_domain=True,
        semantic_encoder=semantic_encoder,
    )


def main():
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    semantic_encoder = SemanticEncoder(cfg["retrieval"]["semantic_encoder"])

    # --- T1, once per seed (Table 2/6/8/10/11 all share these) ---
    for seed in cfg["seeds"]:
        print(f"\n=== T1 training, seed={seed} ===", flush=True)
        split_items = load_and_split(cfg, seed, semantic_encoder)
        train_items = [it for it in split_items if it.split == "train"]
        val_items = [it for it in split_items if it.split == "val"]

        checkpoint_path = f"{cfg['paths']['t1_checkpoint_dir']}/seed{seed}"
        if SupervisedTransformerDetector.has_checkpoint(checkpoint_path):
            print(f"checkpoint already exists at {checkpoint_path}, skipping", flush=True)
            continue

        t1_cfg = T1Config(**{**cfg["t1_supervised"], "seed": seed})
        t1 = SupervisedTransformerDetector(t1_cfg)
        t1.fit(train_items, val_items)
        t1.save(checkpoint_path)
        print(f"saved T1 checkpoint to {checkpoint_path}", flush=True)

    # --- Quick single-config domain classifier (routing context for Tables
    # 6/8/10/11), once per seed. Mirrors experiments.common's
    # build_routing_domain_classifier, duplicated here rather than imported
    # since that module pulls in rank_bm25/openai, which this notebook
    # deliberately doesn't install. ---
    for seed in cfg["seeds"]:
        print(f"\n=== quick domain classifier, seed={seed} ===", flush=True)
        split_items = load_and_split(cfg, seed, semantic_encoder)
        train_items = [it for it in split_items if it.split == "train"]
        val_items = [it for it in split_items if it.split == "val"]

        checkpoint_path = f"{cfg['paths']['quick_domain_classifier_checkpoint_dir']}/seed{seed}"
        import os

        if os.path.exists(os.path.join(checkpoint_path, "wrapper_meta.json")):
            print(f"checkpoint already exists at {checkpoint_path}, skipping", flush=True)
            continue

        dc_cfg = cfg["domain_classifier"]
        dom_cfg = TrainConfig(
            model_name=dc_cfg["candidate_encoders"][0],
            lr=dc_cfg["lr_grid"][0],
            batch_size=dc_cfg["batch_size_grid"][0],
            max_length=dc_cfg["max_length_grid"][0],
            seed=seed,
        )
        domain_model, _ = train_one_config(train_items, val_items, dom_cfg)
        domain_model.save(checkpoint_path)
        print(f"saved quick domain classifier checkpoint to {checkpoint_path}", flush=True)

    # --- Domain classifier grid search (Table 5), single seed ---
    seed = cfg["seeds"][0]
    split_items = load_and_split(cfg, seed, semantic_encoder)
    train_items = [it for it in split_items if it.split == "train"]
    val_items = [it for it in split_items if it.split == "val"]

    for encoder_name in cfg["domain_classifier"]["candidate_encoders"]:
        print(f"\n=== domain classifier grid search: {encoder_name} ===", flush=True)
        grid_search_domain_classifier(
            [encoder_name],
            train_items,
            val_items,
            lr_grid=cfg["domain_classifier"]["lr_grid"],
            batch_size_grid=cfg["domain_classifier"]["batch_size_grid"],
            max_len_grid=cfg["domain_classifier"]["max_length_grid"],
            seed=seed,
            checkpoint_dir=cfg["paths"]["domain_classifier_checkpoint_dir"],
        )
        # The winning model for this encoder is already saved to disk and its
        # return value isn't kept here -- explicitly release its GPU memory
        # (on top of grid_search_domain_classifier's own per-config cleanup)
        # before the next, possibly larger, encoder starts training.
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll done. Zip and download the checkpoints/ folder next.", flush=True)


if __name__ == "__main__":
    main()
