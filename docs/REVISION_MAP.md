# Reviewer points → code → outputs

How each item of the revision plan is implemented and where its evidence is produced.

| Reviewer point | Implementation | Output |
|---|---|---|
| 1.1 System-level component ablation (Planner, Actor, Preprocessor, Analyzer, Recorder, Alignment) | `experiments/configs.py` (F, B0, A1–A6), `pipeline.py` switches, `experiments/run_ablation.py` | `results/tables/ablation_main.*`, `table_ablation_full_metrics.*`, `fig_ablation_forest` |
| 1.1 "Is it more than a good prompt?" | **B0** = direct LLM + Fixed Instruction Baseline | same tables |
| 1.2 What does "task-adaptive" adapt; what replaces the removed prompt | `S_D` in `schema.TaskSpecification`; `P_{D,M} = R_M(C(S_D))` in `actor.py`; `prompts/fixed_instruction_baseline.txt` (FIB) for A2/B0; `docs/ABLATION_DESIGN.md` | `task_specification.json`, `prompt_bundle.json` per run (release as Supplementary) |
| 1.3 CIs, significance, multiple runs | `evaluation/stats.py`: paired bootstrap (10,000), exact McNemar, Holm; 3 runs pooled; `multirun_stability` | `ablation_main.*` (ΔAcc, ΔF1, 95% CI, p, p_adj), `multirun_stability.*` |
| 1.4 Split eLife / SubstanReview | every table is per dataset; pooled block sums counts; dataset contrast ΔF1(eLife) − ΔF1(SR) | `ablation_main.*`, `ablation_contrast.*` |
| 1.5 Move ablation before robustness | tables are generated independently; suggested order: Main → Component contribution → Robustness (multi-run, R1/R2, confusion) → Scientometrics | `table_performance`, `ablation_main`, `multirun_stability`, `reflection_r1_r2`, `confusion.csv` |
| 2.1 / 2.2 Users, scenarios, research gap | text-only; the scientometrics module is the downstream use-case evidence | `scientometrics/` |
| 3.1 Alignment metric contradiction | alignment now enters span P/R/F1: rejected spans are false positives (`evaluation/metrics.py`, `rejected_span_policy`); A5 = exact matching only; new metrics: exact span match, token IoU, recoverable / rejected span rate | `table_alignment.*`, `ablation_main` row A5 |
| 3.1 τ definition | `TP = 1[y_p = y_g ∧ IoU_tok ≥ τ]`, τ = 0.5, sensitivity grid | `table_tau_sensitivity.*`, `fig_tau_sensitivity` |
| 3.2 Numerical audit from raw predictions | `experiments/evaluate.py` (single source), `evaluation/audit.py` (F1 identity, counts, Accuracy = Recall diagnosis, pooled vs mean, manuscript cross-check) | `results/master/*.csv`, `audit_report.txt` |
| 3.2 "Average" → "Pooled across datasets" | `make_tables.py` labels the block *Pooled across datasets* and reports the underlying counts | `table_performance*.md` |
| 3.2 Transformers Accuracy = Recall | baselines produce unit-level predictions; audit explains the structural identity when it holds; counts N, N_correct, TP, FP, FN are tabulated | `table_performance_counts.*`, audit INFO lines |
| 3.3 / 3.4 Diagnostic composite score, Figure 6 | removed; R1→R2 is a plain delta table/plot with bootstrap CIs; Gemini Major_Claim wording follows the per-label table | `reflection_r1_r2.*`, `fig_reflection_delta` |
| 4.1 Algorithm pseudocode | `docs/ALGORITHMS.md` (Algorithm 1 pipeline, Algorithm 2 alignment); Summarizer renamed **Preprocessor** (`preprocessor.py`) | – |
| Figures 5–7 | one clean figure per message (`make_figures.py`, ≥ 8–9 pt at column width, hatching not colour-only) | `results/figures/` |
| Reproducibility subsection | `docs/REPRODUCIBILITY.md`; manifests; exact model ids; cache of raw outputs; `--check-model` | `manifest.json` per run |
| Range Length → token-range metrics, N/A for transformers | `make_tables.py` prints `N/A` when `alignment_applicable` is false | `table_performance.*` |
| Reliability (κ, ICC) | `evaluation/agreement.py` (`cohen_kappa`, `icc_oneway`, `fleiss_kappa`, `inter_run_agreement`); SubstanReview IAA files loaded by `data/substanreview.load_iaa` | – |
| RQ3 downstream utility | `scientometrics/`: features (Section 4.3.4 formulas), OpenAlex 5-year citations, NB / Spearman-bootstrap / quantile / Mann–Whitney / k-means / LASSO-EN | `results/scientometrics/` |

## Facts surfaced by the code that the text should reflect

* The public SubstanReview release contains a `Major_claim` label (163 spans = Table 4).
* OpenRouter/Poe no longer serve Claude-3.5-Sonnet, GPT-4o-Latest or Gemini-1.5-Pro (2026-09);
  the revision must state the exact snapshots used (`models_returned` in the manifests).
* With unit-level baselines, Recall = Accuracy holds *by construction* only when predicted spans
  coincide with the evaluated units; with sentence-unit baselines evaluated under the τ criterion
  it need not hold – the audit reports which case applies.
