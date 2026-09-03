# Implementation and reproducibility (manuscript Section 4.X)

Every run directory `results/raw/<dataset>/<backend>/<config>/run<k>/` contains:

| Category | Recorded in | Fields |
|---|---|---|
| LLM | `manifest.json` | provider, `model_requested`, `models_returned` (ids reported by the provider for every call), backend family |
| API | `manifest.json` | `started_at`, `finished_at` (UTC), elapsed time, number of calls, cache hits, token usage |
| Sampling | `manifest.json` / `configs/experiment.yaml` | temperature 0.0, top_p 1.0, max_tokens 4096, seed (+ run index where the provider supports seeds) |
| Repetition | run directories | 3 runs (`run0..run2`), pooled or single-run analysis (`--runs-policy`) |
| Prompt | `prompt_bundle.json`, `task_specification.json` | full system prompt, user template, prompt hash, specification hash, synthesis mode (deterministic / Table 2 meta-prompt), FIB text |
| Validation | `manifest.json`, `predictions.jsonl` | retry budget R = 2, per-review retries, validation issues per attempt, flagged reviews |
| Alignment | `manifest.json` | tokenizer, similarity threshold 0.80, normalisation, boundary snapping, tie-breaking rule |
| Dataset | `manifest.json`, `data/*/manifest_*.json` | exact record ids, SHA-256 of each text, dataset hash |
| Transformers | baseline `manifest.json` | checkpoint, seed, optimizer/scheduler, batch sizes, epochs, early stopping, training history |
| Evaluation | `results/master/evaluation_settings.json` | τ grid, primary τ, rejected-span policy, tokenizer, unit mode |
| Statistics | `configs/experiment.yaml` | 10,000 bootstrap resamples, seed 12345, 95% CI, Holm |
| Code | `manifest.json` | package version, git commit |

Raw LLM responses are archived in the SQLite cache (`results/cache/llm_cache.sqlite`, keyed by
provider, model, messages, parameters and run index).  Re-running the evaluation never calls an API.

Suggested data-availability statement: *Code, prompt templates (including the Fixed Instruction
Baseline and the generated adaptive prompts), processed evaluation manifests, annotation
guidelines, raw model outputs and the metric/statistics scripts are available at
<https://github.com/psknlr/ZengziAgent> (archived on Zenodo upon publication).*
