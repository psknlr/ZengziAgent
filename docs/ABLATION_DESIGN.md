# Component-contribution and ablation design (revised Section 6)

The multi-stage framework is evaluated with a *system-level* ablation matrix.  Every removed
component has an explicit replacement, so each contrast is a well-defined pair of systems run on
the **same test cases** (paired design).

| ID | Configuration | Planner | Adaptive prompt | Preprocess | Validation | Refinement | Alignment | Replacement of the removed component |
|---|---|---|---|---|---|---|---|---|
| F  | Full ZengziAgent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| B0 | Direct LLM + Fixed Instruction Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | one call with `P_fixed`; raw text; tolerant single parse; verbatim match only |
| A1 | w/o Planner | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | generic specification `S_generic` (label names + one-line definitions, objective, output format); no dataset profile, no demonstration selection, no dataset-specific constraints |
| A2 | w/o Task-Adaptive Prompting | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | `P_fixed` (FIB) instead of `P_adaptive(S_D, M) = R_M(C(S_D))`; the Planner output is still used by the validator (allowed labels, rules) |
| A3 | w/o deterministic preprocessing | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | identity mapping: raw text is the prompt input and the alignment reference |
| A4 | w/o Recorder validation/refinement | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | first response parsed tolerantly; illegal labels and unrecoverable spans are kept as predictions (false positives) |
| A5 | w/o Text Alignment | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | exact string matching only: normalised matching, fuzzy recovery, boundary correction and token-range remapping disabled; non-verbatim spans rejected (false positives); validation still flags them and may trigger refinement |
| A6 | validation only (w/o refinement retry) | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | validation without re-query (R = 0): invalid items are flagged/dropped, never repaired |

`B0` answers the question *"how much better is the framework than giving the same LLM a good
fixed prompt?"*; `A2` vs `B0` isolates the non-prompt stages (preprocessing + validation +
alignment) under an identical prompt.

## Fixed Instruction Baseline (FIB)

`prompts/fixed_instruction_baseline.txt` – dataset-independent; contains only the annotation
objective, the permitted labels and the output format.  No dataset profile, no examples, no
context-specific instructions, no backend-specific rendering.  It is the `P_fixed` of the
manuscript and is released verbatim in the Supplementary Material.

## Task-adaptive prompting, formally

```
S_D      = (U_D, L_D, G_D, E_D, C_D, O_D, V_D)        Planner output (configs/schema + corpus profile)
P_{D,M}  = R_M( C(S_D) )                              compiler C -> prompt IR; renderer R_M per backend family
```
Task adaptation is the **deterministic compilation of an externally defined annotation
specification** into dataset- and backend-conditioned prompts; it involves no online learning and
no parameter update.  The specification and the rendered prompt are written next to every run
(`task_specification.json`, `prompt_bundle.json`) and hashed into the manifest.

## Statistics

* **ΔF1, ΔAccuracy** with **paired bootstrap** CIs: reviews are resampled with replacement
  (10,000 resamples), counts are summed and micro-F1 / accuracy recomputed for both systems from the
  same resample; percentile 95% CI; two-sided bootstrap p.
* **McNemar (exact binomial)** on paired unit-level correctness; with pooled runs each *distinct* unit contributes one pair (correct = correct in at least half of the runs), so repeated runs never inflate the discordant counts.
* **Holm** step-down correction within each dataset × backend family of ablation contrasts
  (7 contrasts) and within the family of pairwise backend comparisons.
* **Dataset contrast** `ΔF1(eLife) − ΔF1(SubstanReview)` with independent review-level
  resampling inside each dataset – answers *which corpus depends more on each component*.
* Repeated runs: counts are pooled over the 3 runs by default (`--runs-policy pooled`) so that all
  runs contribute; `--runs-policy run0` reproduces a single-run analysis.  Multi-run stability
  (mean ± SD, inter-run label agreement, Fleiss' κ) is reported separately.

Outputs: `results/tables/ablation_main.{csv,md}` (main table: Dataset, Backend, Ablation,
ΔAccuracy, ΔF1, 95% CI, p, adjusted p; `n_units` = distinct evaluated units, `n_unit_run_pairs` = units × runs entering the pooled counts), `ablation_contrast`, `backend_pairwise`,
`multirun_stability`, `reflection_r1_r2`, and `table_ablation_full_metrics` (Supplementary P/R/F1).
