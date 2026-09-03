# results/

| Path | Content | Committed? |
|---|---|---|
| `raw/<dataset>/<backend>/<config>/run<k>/` | raw predictions, manifests, prompts, specifications of real experiments | no (regenerate with `scripts/run_all.sh`) |
| `cache/llm_cache.sqlite` | archive of every raw LLM response | no |
| `master/` | single-source evaluation tables (`master_results.csv`, `per_label.csv`, `review_counts.csv`, `unit_predictions.csv`, `confusion.csv`) | no |
| `tables/`, `figures/` | manuscript tables (CSV/Markdown/LaTeX) and figures (PNG/PDF) generated from `master/` | produced by the pipeline |
| `mock_smoke/tables`, `mock_smoke/figures` | example outputs of the **offline mock backend** (`scripts/run_mock_smoke.sh`) – they demonstrate the format only and carry **no scientific meaning** | yes (small) |

Never edit a number in `tables/` by hand: change the raw predictions or the evaluation code and
re-run `evaluate → analyze_ablation → make_tables → make_figures → audit`.
