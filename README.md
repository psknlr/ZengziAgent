# ZengziAgent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/psknlr/ZengziAgent/blob/claude/paper-revision-ablation-study-dgwirx/notebooks/ZengziAgent_Colab.ipynb)

Code for **ZengziAgent: a schema-conditioned multi-stage LLM framework for fine-grained scientific
peer-review annotation** (revision for *Scientometrics*).  The repository contains the complete
annotation pipeline, the component-contribution (ablation) study, the transformer baselines, the
single-source evaluation/statistics scripts that generate every table and figure, and the
scientometric application.

```
Schema (labels, guideline, examples)
   │  Stage 1  Planner            S_D = (U_D, L_D, G_D, E_D, C_D, O_D, V_D)
   ▼
Structured task specification
   │  Stage 2  Actor              P_{D,M} = R_M(C(S_D))   (compiler C, backend renderer R_M)
   ▼
Executable prompt ──► Stage 3 Preprocessor (deterministic, offset-preserving)  x → x'
   │
   ▼  Stage 4  Analyzer           schema-constrained XML annotation with an LLM backend
   │
   ▼  Stage 5  Recorder           XML validity · label legality · non-empty spans · recoverability
   │            └── bounded refinement loop (≤ R re-queries with the validation error)
   ▼
Text Alignment Tool (Algorithm 2)  exact → normalised → fuzzy (difflib) → token boundaries; reject below τ_align
   │
   ▼
Validated, source-aligned annotations  →  evaluation (unit accuracy, span P/R/F1 with token IoU ≥ τ)
```

## 1. Installation

```bash
git clone https://github.com/psknlr/ZengziAgent && cd ZengziAgent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core + statistics + figures + scientometrics
pip install torch transformers             # optional: transformer baselines (Table 7)
python -m pytest                           # unit tests (offline; no API key needed)
```

Python ≥ 3.10.  spaCy is used for token-boundary mapping (`spacy.blank("en")`, no model download
needed); a regex tokenizer is the fallback and the manifest records which one was used.

## 2. LLM backends: OpenRouter, Poe, MiniMax

All providers are accessed through the OpenAI-compatible chat-completions protocol
(`zengziagent/llm/openai_compat.py`).  Set one key:

| Provider | Environment variable | Base URL | Backend specs |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `openrouter:claude`, `openrouter:gpt4o`, `openrouter:gemini`, `openrouter:minimax` |
| Poe | `POE_API_KEY` | `https://api.poe.com/v1` | `poe:claude`, `poe:gpt4o`, `poe:gemini`, `poe:minimax` |
| MiniMax | `MINIMAX_API_KEY` | `https://api.minimax.io/v1` (`minimax-cn:` → `api.minimaxi.com`) | `minimax:minimax` |
| any | as above | – | `provider:model=<exact model id>` |

Aliases are defined in `configs/backends.yaml`.  **Model snapshots:** the original submission used
Claude-3.5-Sonnet, GPT-4o-Latest and Gemini-1.5-Pro; as of 2026-09 OpenRouter/Poe no longer list
those ids, so the aliases point to dated/current snapshots and the run manifest records both the
requested id and the id returned by the provider.  Use `--check-model` to verify an id against the
provider catalogue before a long run, or pass `provider:model=<id>` explicitly.

`mock` / `mock:noisy` is a deterministic offline backend used only for tests and smoke runs. `bash scripts/run_all.sh <provider>` and the notebook derive the backend list from `configs/backends.yaml` (the three paper backends where the provider serves them, otherwise every alias it serves, e.g. `minimax:minimax`).

## 3. Reproducing the experiments

```bash
python scripts/prepare_data.py --substanreview      # downloads SubstanReview (440/110 reviews) + manifests
bash scripts/run_all.sh openrouter                  # whole protocol (see below); idempotent, cached
bash scripts/run_mock_smoke.sh                      # offline end-to-end check (~3 min, no keys)
```

Step by step (each command has `--help`):

| Step | Command | Output |
|---|---|---|
| Main results + ablations | `python -m zengziagent.experiments.run_ablation --backends openrouter:claude openrouter:gpt4o openrouter:gemini --runs 3` | `results/raw/<dataset>/<backend>/<config>/run<k>/predictions.jsonl` + `manifest.json` + `prompt_bundle.json` + `task_specification.json` |
| One configuration | `python -m zengziagent.experiments.run_annotation --dataset elife --backend poe:gpt4o --configs F A2 A5 --runs 3` | as above |
| Reflection R2 | `python -m zengziagent.experiments.run_reflection --backend openrouter:claude --configs F` | `.../run<k>/round2/predictions.jsonl` |
| Baselines | `python -m zengziagent.baselines.transformers_baseline --dataset substanreview --model roberta --seeds 13 42 2024` | `results/raw/<dataset>/hf-roberta/TB/run<seed>/` |
| **Evaluate (single source of truth)** | `python -m zengziagent.experiments.evaluate` | `results/master/master_results.csv`, `per_label.csv`, `review_counts.csv`, `unit_predictions.csv`, `confusion.csv` |
| Statistics | `python -m zengziagent.experiments.analyze_ablation` | `results/tables/ablation_main.*`, `ablation_contrast.*`, `backend_pairwise.*`, `multirun_stability.*`, `reflection_r1_r2.*` |
| Tables / figures | `python -m zengziagent.experiments.make_tables` · `make_figures` | `results/tables/table_*.{csv,md,tex}`, `results/figures/fig_*.{png,pdf}` |
| Audit | `python -m zengziagent.evaluation.audit results/master/master_results.csv [--manuscript-csv table11.csv]` | consistency report (F1 = 2PR/(P+R), counts, Accuracy = Recall diagnosis, pooled vs mean) |
| Scientometrics | `python -m zengziagent.scientometrics.build_features ...` · `analysis` | `results/scientometrics/*.csv|json`, `fig_scientometrics.png` |

Runs are idempotent: a run directory is reused only when its manifest lists the same record ids and no errors; runners exit non-zero when any review errored (re-run to complete it, successful calls are cached). Non-retryable provider errors (bad key, unknown model) abort immediately.

The `zengzi` console script (requires an **editable** install of the checkout, `pip install -e .`, because `configs/` and `prompts/` are read from the repository; alternatively set `ZENGZI_ROOT=/path/to/checkout`) exposes the same commands
(`zengzi annotate|ablate|reflect|evaluate|analyze|tables|figures|audit|baseline|scientometrics`).

## 3b. Fast reproduction in Google Colab

`notebooks/ZengziAgent_Colab.ipynb` runs the reviewer-requested experiments end to end in one
notebook: clone + install, API key from Colab Secrets (OpenRouter / Poe / MiniMax), data download,
component-contribution matrix (F, B0, A1–A6) with repeated runs, R1→R2 reflection, single-source
evaluation, paired-bootstrap/McNemar/Holm statistics, tables, figures, audit, optional GPU
baselines and scientometrics, and a zip download of every artefact.  `FAST = True` (default)
uses 30 reviews × 2 runs × 2,000 resamples per backend (≈ 20–40 min for one backend);
`FAST = False` runs the full protocol.  `PROVIDER = "mock"` (or the environment variable
`ZENGZI_PROVIDER=mock`) exercises the notebook offline without a key.

The notebook is generated by `scripts/build_colab_notebook.py`; `scripts/test_notebook_offline.py`
executes its code cells offline (mock provider) as a smoke test.

## 4. Component-contribution study (ablations)

| ID | Configuration | What replaces the removed component |
|---|---|---|
| F | Full ZengziAgent | – |
| B0 | Direct LLM + Fixed Instruction Baseline | static prompt `prompts/fixed_instruction_baseline.txt`, raw text, single pass, verbatim matching only |
| A1 | w/o Planner | generic specification (labels + format only; no dataset profile, demonstrations or dataset constraints) |
| A2 | w/o Task-Adaptive Prompting | `P_fixed` (FIB) instead of `P_adaptive(S_D, M)`; other stages unchanged |
| A3 | w/o deterministic preprocessing | identity preprocessing (raw text is prompt input and alignment reference) |
| A4 | w/o Recorder validation/refinement | single-pass tolerant parsing; illegal labels / unrecoverable spans kept as (false-positive) predictions |
| A5 | w/o Text Alignment | exact string matching only; non-verbatim spans rejected and counted as false positives |
| A6 | validation only | validation without re-query (R = 0) |

Every ablation is reported per dataset (eLife, SubstanReview-derived) and per backend with
ΔAccuracy, ΔF1, paired-bootstrap 95% CIs (10,000 review-level resamples), bootstrap and exact
McNemar p-values, Holm-adjusted within each dataset × backend family, and the dataset contrast
ΔF1(eLife) − ΔF1(SubstanReview).  See `docs/ABLATION_DESIGN.md`.

## 5. Metrics (Section 4.3.3)

* **Unit-level accuracy** – each gold span is an evaluated unit with one primary label; the predicted
  label is that of the recovered predicted span with the largest token overlap (`None` if none).
* **Span-level micro P/R/F1** – `TP = 1[y_p = y_g ∧ IoU_tok ≥ τ]`, τ = 0.5 (sensitivity over
  {0.3, 0.5, 0.7, 0.9, 1.0} reported), one-to-one greedy matching; rejected (unrecoverable) spans
  count as false positives; *Pooled across datasets* sums the counts.
* **Alignment metrics** – exact span match, mean token IoU, recoverable span rate, rejected span
  rate; `N/A` (not 0) for baselines whose output mechanism has no free-form source spans.

`docs/REPRODUCIBILITY.md` lists what every run manifest records (model ids, sampling parameters,
dates, prompt/spec/dataset hashes, retries, tokenizer, alignment threshold, git commit).

## 6. Repository layout

```
zengziagent/
  schema.py            data structures: S_D, PipelineConfig, ReviewRecord, GoldSpan, AnnotationUnit
  planner.py           Stage 1  (requirement analysis, dataset characterisation, demonstration selection)
  actor.py             Stage 2  (compiler C, renderer R_M, optional LLM prompt synthesis, FIB)
  preprocessor.py      Stage 3  (offset-preserving normalisation, sentence segmentation)
  analyzer.py          Stage 4  (LLM call, tolerant XML parsing)
  recorder.py          Stage 5  (validation, refinement feedback, final record)
  alignment.py         Text Alignment Tool (Algorithm 2)
  pipeline.py          Algorithm 1 with ablation switches
  tokenization.py      spaCy / regex tokenisers, token IoU
  llm/                 OpenAI-compatible client (OpenRouter, Poe, MiniMax, …), SQLite cache, mock backend
  data/                SubstanReview loader + label mapping, eLife XML extraction, manifests
  evaluation/          metrics, statistics (bootstrap, McNemar, Holm), agreement (κ, ICC), audit
  experiments/         runners, ablation presets, evaluate, analyze, tables, figures
  baselines/           RoBERTa / ELECTRA / ALBERT / GPT-2 fine-tuning (Table 7)
  scientometrics/      review features, OpenAlex citations, NB / quantile / MWU / k-means / LASSO
configs/               experiment.yaml (frozen protocol), backends.yaml, schema/five_label.yaml
prompts/               fixed_instruction_baseline.txt (Supplementary), actor_meta_prompt.txt (Table 2), reflection_prompt.txt
docs/                  ABLATION_DESIGN, ALGORITHMS, DATA, REPRODUCIBILITY, REVISION_MAP
scripts/               prepare_data.py, run_all.sh, run_mock_smoke.sh
tests/                 pytest suite
results/mock_smoke/    example tables/figures produced by the offline mock (no scientific meaning)
```

## 7. Data

* **SubstanReview** (Guo et al., 2023; Apache-2.0) is downloaded from the public repository; labels
  `Eval_*/Jus_*` are used as released and `Major_claim` is mapped to `Major_Claim`
  (`docs/DATA.md` documents the mapping and the fact that the release already contains 163
  `Major_claim` spans).
* **eLife** reviewer text is extracted from `elife-article-xml`; the human gold annotations of the
  study are a separate JSONL file (same schema) joined by `review_id`.

## Citation

Kang, Y., Zhang, X., Chen, Y., & Sun, Z. ZengziAgent: a schema-conditioned multi-stage LLM framework for
fine-grained scientific peer-review annotation. *Scientometrics* (under revision).
