# Frozen Experimental Protocol

Status: DRAFT for approval. Once approved, nothing below changes without a dated entry in the deviation log at the bottom.

## 1. Research questions

1. Does a router that selects among detectors beat the best single detector?
2. Does affective retrieval beat semantic retrieval for the retrieval-augmented language model detector?
3. Do the answers hold when a whole domain is unseen (leave-one-domain-out) and on a second dataset?
4. Where does the headroom lie (the gap between the oracle and the best single detector), and why do routers fail to capture it?

## 2. Data

- Primary: AMTCele, 980 items (six domains of 80 items each, plus celebrity with 500).
- Generalization: PHEME and/or COCO (code exists; downloads required). Added once the primary study is complete.

## 3. Evaluation design

- Repeated stratified five-fold cross-validation over all 980 items, so every item is tested exactly once per repetition. Repetitions use seeds 13, 42 and 2024 (different fold assignments).
- Folds are stratified by domain and label. Exact duplicates and near-duplicates (cosine similarity of 0.95 or higher) always stay in the same fold.
- Inside each outer training portion (80 percent of the data), 75 percent is training and 25 percent is validation (so validation is 20 percent of all data, about 196 items). The test fold is 20 percent.
- Everything that is chosen is chosen on the validation portion only: prompt versions, thresholds, fusion weights, the performance catalogue. The test fold is used only for final reporting.
- Per-domain catalogue cells will still be small (about 16 items in the six small domains). Catalogue entries therefore report their item counts and are shrunk toward the all-domain value when counts are small. Primary routers rely mainly on pooled statistics; per-domain cells are secondary.

## 4. Leakage rules (mandatory)

1. A retrieved example for item X must never be X itself or any near-duplicate of X. This applies to validation items as well as test items. (A violation was found on 2026-10-16, see the deviation log.)
2. Test items never appear in any retrieval pool, training set, catalogue or fusion-weight fit.
3. The domain classifier is trained on the training portion only.
4. Cached predictions are keyed by seed and fold; nothing is shared across folds when the retrieval pool differs.

## 5. Systems compared

Detectors: the supervised transformer (fine-tuned RoBERTa-base), the retrieval-augmented language model (affective-semantic retrieval, 4 examples, alpha 0.5), and the evidence verifier (BM25 keyword retrieval over the Wikipedia entity corpus). Language model: qwen/qwen3.5-9b served locally, temperature 0, reasoning disabled.

Baselines: majority class; language model zero-shot; random examples; semantic retrieval; affective retrieval; confidence-weighted ensemble of all three detectors; domain-rule router; oracle. The all-detector language model advisor is run on a subset only (about 34 seconds per call).

Router variants (fixed list, no additions after approval):
1. Original-prompt router.
2. Router without the evidence-verifier rule.
3. Router without the evidence-verifier rule and without retrieved-example metadata.
4. Soft-fusion combiner: blends all three detectors' fake-probabilities using weights from validation performance and calibration. No extra language model calls.
5. Supervised-transformer-first with escalation: uses the supervised transformer unless its confidence is below a threshold chosen on validation; then consults the other detectors. No extra language model calls.

## 6. Metrics

Primary: macro-averaged F1. Secondary: accuracy, area under the receiver operating characteristic curve, expected calibration error, routing accuracy, detector calls per item, latency.

## 7. Statistics

- Paired bootstrap with 10,000 resamples over pooled out-of-fold predictions; report 95 percent confidence intervals.
- Holm correction across the pre-registered comparison family: (a) each router versus the best single detector, (b) affective versus semantic retrieval, (c) soft fusion and escalation versus the confidence-weighted ensemble.
- Report the spread across the three seeds.

## 8. Reporting rules

- Anything not listed here is labeled exploratory in the paper.
- If a router beats the best single detector with a confidence interval excluding zero, lead with it. If not, the paper is the diagnostic study. Both outcomes are reported with the same tables.

## 9. Out-of-domain evaluation

Leave-one-domain-out with a nearest-neighbour domain proxy (the closed-set classifier cannot name an unseen domain; its 0 percent accuracy is reported as a limitation).

## 10. Known issues log

| Date | Issue | Consequence |
|---|---|---|
| 2026-10-16 | Validation items retrieve themselves (147 of 147 had their own text as the top example, with its gold label), inflating the retrieval-augmented detector's validation score (0.939 on validation versus 0.752 on test). | The performance catalogue and every router result computed from it (Tables 2, 6, 7, 8, 10, 11) are invalid and must be recomputed after the fix. |
| earlier | Router ablations (Table 10) and the removal of the evidence-verifier rule were chosen after viewing test results. | Redo selection on validation data only. |
| 2026-11-05 | Leak fixed (each retrieval excludes the query and its near-duplicates, cosine 0.95 or higher). Catalogue, Table 6, fusion and significance recomputed; old outputs kept as `*.leaked.json`. | Resolved. |
| 2026-11-08 | Router variants 1 to 3 rerun on validation and test with the clean catalogue (`results/router_variants.json`). Validation macro F1: original prompt 0.816 (chose the evidence verifier for 64 of 147 items), without the evidence-verifier rule 0.871, also without retrieved-example metadata 0.871. | Removal of the evidence-verifier rule is now supported on validation alone (ties go to the variant listed first). Resolves the "chosen after viewing test results" issue for the prompt. |
| 2026-11-08 | Table 11 counted only the chosen tool's time for the proposed router, not the router's own language model call (reported 0.19 seconds per item). Corrected from saved decisions: 6.18 seconds for the routing call plus 0.19 for the chosen tool, 6.37 seconds per item in total (0.12 of the all-tool advisor). Peak graphics memory reads 0 because the language model runs in a separate LM Studio process. | Fixed in `eval/efficiency.py`; rerun `run_efficiency` (all cached, about a minute) after leave-one-domain-out finishes. Report graphics memory from LM Studio separately or drop the column. |
| 2026-11-10 | Clean leave-one-domain-out finished (`results/table7_leave_one_domain_out.json`). Macro-average: always supervised transformer 0.898, domain-rule router 0.875, language model router 0.870, oracle 0.986. Celebrity held out: supervised transformer 0.527, retrieval-augmented detector 0.776; both routers chose the supervised transformer for all 500 (domain classifier labelled 492 entertainment, mean confidence 0.97). Education and business folds lost 14 and 2.5 points from a tie and a one-article difference in 12-item catalogue cells. Table 11 regenerated: router 6.37 seconds per item, matching the corrected value. | The run was interrupted twice; caches made it resume. On the final resume, supervised transformer training was skipped in folds whose predictions were all cached, and the domain classifier was retrained once per fold and its held-out predictions are now saved (`domain_preds_heldout.json`). Router decisions for folds 1 to 6 were made with domain predictions from an earlier, identically seeded training run; the domain-rule router uses the final run's predictions. Table 11's `llm_calls` for the all-tool advisor reads 2 because the advisor's own call is not counted; the true figure is 3 (used in the paper). |
| earlier | Results are single-seed on a 145-item test set; Table 5 still contains the pre-fix DeBERTa entry. | Table 5 DeBERTa entry resolved 2026-11-11 (see next row); single seed remains a limitation. |
| 2026-11-11 | Table 5 rerun: DeBERTa-v3-base retrained (6 settings, length 256), best lr 3e-5, batch 8, validation 0.926, test macro F1 0.928, balanced accuracy 0.915, top-2 0.986, calibration error 0.045. RoBERTa and XLM-RoBERTa rows unchanged (loaded checkpoints). RoBERTa remains the selected encoder, so the neighbour rows and every downstream result are unchanged. | `results/table5_domain_classification.json`; previous version kept as `.before_deberta_rerun.json`. |

## Deviation log

| Date | Deviation | Reason |
|---|---|---|
| 2026-11-05 | Stacked logistic fusion chooses its regularization by validation log loss, not validation macro F1. | The installed scikit-learn raises an error for macro F1 scoring inside LogisticRegressionCV; log loss is a standard proper scoring rule for fitting probabilities. Chosen before seeing test results. |
| 2026-11-11 | Table 5 DeBERTa-v3-base grid searched over maximum length 256 only (6 settings: 3 learning rates by 2 batch sizes), via `configs/table5_deberta_256.yaml`; RoBERTa and XLM-RoBERTa keep their full 12-setting searches (loaded from saved checkpoints). | The old DeBERTa checkpoint (validation 0.871 when saved) predicts a single class when reloaded under transformers 5.14.1 (weights and tokenizer verified identical), so DeBERTa had to be retrained and tested in one run. At length 512 DeBERTa exceeds the 8 GB graphics card and spills into system memory: 12 minutes per epoch against 37 seconds at 256, about 10 to 12 hours for the six 512 settings. The aborted 512 run is in `logs/table5_domain_classification.deberta512_aborted.log` (setting 1 at 256: validation 0.903). Decided before seeing any DeBERTa test result. Broken checkpoint kept at `checkpoints/domain_classifier/microsoft__deberta-v3-base.broken_reload`. |
