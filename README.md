# Affective-Aware Cross-Domain Tool Orchestration for Fake-News Detection

Reference implementation of the framework described in *"Can LLM Agents
Select Better Fake News Detectors? An Empirical Study of Affective-Aware
Cross-Domain Tool Orchestration"* (Nafir draft v0.1).

**Status: no experiments have been run.** This repository is the empirical
framework the paper describes (Sections 3-6 and Appendix A), not a report of
results. Every table script (`experiments/run_*.py`) is fully implemented
and syntax-checked (`python -m py_compile`), but nothing has been executed
end-to-end: there is no installed environment, no downloaded checkpoint, and
no copy of AMTCele/PHEME/COCO in this repo. Filling in the paper's
bracketed `[ ]` results means actually running these scripts against real
data and a real LLM backend and reporting what comes out — do not present
numbers from this codebase as results unless they came from a real run.

## Layout and paper mapping

| Module | Paper section | Purpose |
|---|---|---|
| `data/` | 4.1 | AMTCele / PHEME / COCO loaders |
| `data/splits.py` | 4.2 | Near-duplicate detection, group-aware stratified split, leave-one-group-out |
| `retrieval/` | 3.2 | Semantic + affective encoders, FAISS affective-semantic index |
| `domain/` | 3.3 | Domain classifier (3 input variants), grid search, batch prediction |
| `detectors/` | 3.4 | T1 supervised transformer, T2 retrieved-example LLM, T3 evidence verifier, shared LLM backend |
| `catalogue/` | 3.5 | Validation-only performance catalogue (Table 2) |
| `routing/` | 3.1, 3.6, Appendix A | Utility router, LLM selector prompt/schema, oracle, baselines |
| `eval/` | 4.5 | Metrics, paired bootstrap + Holm correction, routing metrics, efficiency logging |
| `experiments/` | 4-6 | One script per results table |
| `configs/default.yaml` | 4.4 | All hyperparameters/paths in one place |

### Experiment scripts → tables

| Script | Table(s) |
|---|---|
| `experiments/run_catalogue.py` | Table 2 (validation catalogue) |
| `experiments/run_domain_classification.py` | Table 5 (domain classifier) |
| `experiments/run_main_veracity.py` | Table 6 (main results) + RQ1 significance test |
| `experiments/run_leave_one_domain_out.py` | Table 7 (LODO transfer) |
| `experiments/run_routing_analysis.py` | Table 8 (routing quality + selection confusion matrix) |
| `experiments/run_transfer_pheme_coco.py` | Table 9 (PHEME / COCO transfer) |
| `experiments/run_ablations.py` | Table 10 (ablations) |
| `experiments/run_efficiency.py` | Table 11 (efficiency) + lambda sweep |

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` marks `bitsandbytes` as non-Windows-only (4-bit local
inference needs a Linux/CUDA host); on Windows, either run under WSL2 or set
`llm.backend: "api"` in the config to use an OpenAI-compatible endpoint
instead of a local quantized checkpoint.

## Required data and checkpoints (none are included)

- **AMTCele** — set `paths.amtcele_raw` to either a directory of
  `<domain>/{Fake,Legit}/*.txt` files (one article per file, domain names
  matching `utils/types.DOMAINS`) or a manifest CSV (see
  `data/amtcele.py` for accepted column names).
- **PHEME** — set `paths.pheme_raw` to the root of the standard
  `pheme-rnr-dataset` layout: `<event>/{rumours,non-rumours}/<thread_id>/source-tweet/*.json`.
  Used only as a secondary transfer benchmark (Table 9); the loader is
  skipped gracefully (with a `[skip]` message) if the directory is absent.
- **COCO** — set `paths.coco_csv` to a CSV with text/topic/stance/id
  columns (see `data/coco.py` for exact column names and the
  conspiracy/unrelated stance mapping). Also optional/skippable.
- **Evidence corpus for T3** — set `paths.evidence_corpus` to a JSONL file
  of `{"doc_id": ..., "text": ...}` passages that `T3`'s BM25 index is built
  over.
- **LLM backend** — edit `llm:` in `configs/default.yaml`:
  - `backend: "local"` loads `llm.model_name` via `transformers` with 4-bit
    quantization (`BitsAndBytesConfig`). Replace the placeholder
    `REPLACE_WITH_EXACT_CHECKPOINT_AFTER_PILOT` with an actual 7-8B instruct
    checkpoint you have access to, after running the pilot referenced in
    Sec 4.4.
  - `backend: "api"` calls an OpenAI-compatible endpoint instead
    (see `detectors/llm_backend.py::APIBackend`); requires the relevant API
    key/base URL to be available in the environment.

None of these paths exist in this repo — every `run_*.py` script will fail
immediately at data loading until you supply them.

## Running

Each script is a module and takes a `--config` path and optional `--seed`
override (defaults to `cfg["seeds"][0]`, i.e. 13):

```bash
python -m experiments.run_catalogue --config configs/default.yaml
python -m experiments.run_domain_classification --config configs/default.yaml
python -m experiments.run_main_veracity --config configs/default.yaml
python -m experiments.run_leave_one_domain_out --config configs/default.yaml
python -m experiments.run_routing_analysis --config configs/default.yaml
python -m experiments.run_transfer_pheme_coco --config configs/default.yaml
python -m experiments.run_ablations --config configs/default.yaml
python -m experiments.run_efficiency --config configs/default.yaml
```

For the seed-robustness claim in Sec 4.4, rerun the seed-sensitive scripts
(`run_domain_classification`, `run_main_veracity`, `run_leave_one_domain_out`)
once per entry in `cfg["seeds"]` (`--seed 13`, `--seed 42`, `--seed 2024`) and
aggregate across runs yourself — the scripts each report a single seed per
invocation.

All scripts write their table's JSON output to `paths.output_dir`
(`results/` by default) and print a human-readable summary to stdout.

## Validation performed so far

- Every cross-module import was manually cross-referenced against the
  defining module's actual signatures (no dependency-installed runtime
  check was possible in the sandbox this was built in — no `pip` was
  available).
- `python -m py_compile` passes with no errors across the entire package.
- No script has been executed against real data, a real model, or a GPU.
  Treat this as unvalidated at runtime until you run it yourself.

## Known simplifications vs. the paper text

- The affective-semantic similarity `S(x,z) = α·cos_sem + (1-α)·cos_aff` is
  implemented via a FAISS `IndexFlatIP` over weighted-concatenated semantic
  and affective vectors (mathematically equivalent to the paper's
  definition, not a literal two-index fusion).
- T3's evidence verifier abstains (label=0, `prob_fake=0.5`, `failed=True`)
  on any item with no retrieved evidence passages, per the fidelity note in
  Sec 7.3.
- The LLM router's `select()` retries up to `router.max_retries` times on
  schema-invalid output before falling back to the validation catalogue's
  `best_validation_tool` for the predicted domain — this fallback path is
  not separately reported as a metric anywhere; if you want a "fallback
  rate" number for the paper you'll need to add logging for it.
