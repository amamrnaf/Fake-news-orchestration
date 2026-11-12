# Fake-News Detector Orchestration with a Local Language Model

Code, configuration, prompts, protocol and per-item predictions for the paper
*"Can a Local Language Model Agent Select Better Fake-News Detectors? An
Empirical Study of Affective-Aware Detector Orchestration"* (Nafir, Youb and
Hamlich, under review).

A locally served language model (Qwen3.5-9B through LM Studio) chooses, for
each news article, one of three detectors:

| Identifier in the code | Detector |
|---|---|
| `T1` | Supervised transformer: fine-tuned RoBERTa-base classifier |
| `T2` | Retrieval-augmented language model: the same local model, prompted with labelled examples chosen by affective-semantic similarity |
| `T3` | Evidence verifier: the local model judging the claim against BM25-retrieved Wikipedia passages |

The router sees the predicted domain distribution, an emotion profile of the
article, metadata about retrieved labelled examples and a validation-only
performance catalogue. It is compared with single detectors, a domain-rule
router, ensembles, learned fusion (stacking, escalation), an all-tool language
model advisor and an oracle, with paired bootstrap confidence intervals and
Holm correction over a comparison family fixed in advance.

## Main findings

- With an honest validation catalogue, the router selects the supervised
  transformer for 144 of 145 test articles and matches it exactly (macro F1
  0.931) at about 53 times its cost per article. The oracle reaches 0.972.
- The validation catalogue is the only input the router relies on: removing
  it costs 9 points, while removing the emotion profile, the retrieved
  examples or the domain distribution changes nothing.
- When a whole domain is unseen (leave-one-domain-out), both routers fall
  below always using the supervised transformer (0.870 and 0.875 against
  0.898). On held-out celebrity articles the supervised transformer drops to
  0.527 while the retrieval-augmented detector keeps 0.776, yet the router keeps
  the supervised transformer for all 500 articles.
- In this split, affective example retrieval beat semantic retrieval for the
  retrieval-augmented detector (+0.111 macro F1, 95% confidence interval
  [+0.037, +0.186]). AMTCele's fake and legitimate articles come in
  topically matched pairs, so part of this gap may come from semantic
  retrieval showing the opposite-label counterpart as an example.
- Letting validation items retrieve themselves inflated one detector's
  validation score from 0.755 to 0.939 and reversed the routing catalogue. The
  fix and its effect are documented in `PROTOCOL.md`.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | AMTCele loader, near-duplicate-aware stratified split, leave-one-domain-out split (PHEME and COCO loaders are included but unused in the paper) |
| `retrieval/` | Semantic and affective encoders and the affective-semantic index |
| `domain/` | Domain classifier: training, grid search and prediction |
| `detectors/` | The three detectors and the shared language model client |
| `perf_catalogue/` | Validation performance catalogue |
| `routing/` | Language model router and its prompt (`selector_prompt.py`), domain-rule router, oracle and other baselines |
| `eval/` | Metrics, paired bootstrap with Holm correction, routing and cost measurement |
| `experiments/` | One script per result (see below) |
| `configs/` | All hyperparameters and paths |
| `results/` | Result tables (JSON) and per-item predictions for every experiment, in `results/prediction_cache/` |
| `PROTOCOL.md` | Frozen evaluation protocol, known-issues log and deviation log |

## Experiments

| Script | Result |
|---|---|
| `experiments/run_catalogue.py` | Validation performance catalogue |
| `experiments/run_domain_classification.py` | Domain classifier comparison |
| `experiments/run_main_veracity.py` | Main veracity results (single detectors, combinations, routers, oracle) |
| `experiments/run_fusion.py` | Learned fusion: performance-weighted fusion, stacking, escalation |
| `experiments/run_router_variants.py` | The three prompt variants of the router, selected on validation |
| `experiments/run_routing_analysis.py` | Routing quality and selection confusion matrix |
| `experiments/run_ablations.py` | Router input ablations |
| `experiments/run_efficiency.py` | Cost per article |
| `experiments/run_leave_one_domain_out.py` | Leave-one-domain-out evaluation |
| `experiments/run_significance.py` | Paired bootstrap tests with Holm correction over the full comparison family |

Every script takes `--config` (default experiments use `configs/default.yaml`)
and an optional `--seed` (default 13), and writes its output to `results/`.
`scripts/run_remaining.cmd` shows the order used on Windows.

All language model and detector predictions are cached per item, write-through,
in `results/prediction_cache/`. Rerunning any script with these caches in place
reproduces the reported numbers without calling the language model again;
delete a cache file to recompute it.

## Installation

Tested with Python 3.13.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Language model

The paper uses `qwen/qwen3.5-9b` served by [LM Studio](https://lmstudio.ai)
through its OpenAI-compatible server at `http://127.0.0.1:1234/v1`, with
temperature 0. Load the model in LM Studio (or run `lms load qwen/qwen3.5-9b`)
before running any script that calls the language model. Any other
OpenAI-compatible server works by editing the `llm:` section of the config.
Use the `qwen/`-prefixed model identifier: it is the variant with reasoning
output disabled.

### Data (not included)

AMTCele combines the FakeNewsAMT and Celebrity datasets of Pérez-Rosas et
al., "Automatic Detection of Fake News", COLING 2018. Request them from the
authors (University of Michigan, Language and Information Technologies group)
and place them as:

```
data_raw/amtcele/fakeNewsDataset/{fake,legit}/<domain><number>.{fake,legit}.txt
data_raw/amtcele/celebrityDataset/{fake,legit}/*.txt
```

The evidence verifier's Wikipedia corpus is built with
`python scripts/build_evidence_corpus.py` (see the script for options);
`data_raw/evidence_corpus.example.jsonl` shows the format.

### Hardware

Results in the paper were produced on a laptop with an 8 GB NVIDIA RTX 4070
graphics card. The language model needs about 6.5 GB of graphics memory, so
the scripts unload it from LM Studio while transformer models train and reload
it afterwards (`utils/lm_studio.py`). Trained checkpoints are not included;
they are retrained when missing.

## Citation

If you use this code, please cite the paper (citation details will be added
on publication).

## Licence

Code released for research use. The AMTCele data belong to their original
authors and are subject to their terms.
