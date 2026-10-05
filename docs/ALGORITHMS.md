# Algorithms

## Algorithm 1 – ZengziAgent annotation pipeline (`zengziagent/pipeline.py`)

```
Input : reviewer text x, annotation schema G, demonstrations E, backend model M,
        retry budget R, alignment threshold τ_align
Output: validated source-aligned annotations A*

1.  S ← Planner(G, E, metadata(x))              # structured task specification S_D
2.  P ← Actor(S, M) = R_M(C(S))                 # compiled, backend-rendered prompt
3.  x' ← DeterministicPreprocess(x)             # offset-preserving; keeps map x' → x
4.  A ← Analyzer(P, x')                         # XML annotation blocks from the LLM
5.  V ← Validate(A, S, x')                      # XML validity, label legality, non-empty spans,
                                                # recoverability (via Algorithm 2)
6.  retry ← 0
    while V has blocking issues and retry < R:
        A ← Analyzer(P, x', validation_error = V) ; V ← Validate(A, S, x') ; retry ← retry + 1
7.  for each predicted span a ∈ A:
        a* ← AlignToSource(a, x', τ_align)      # Algorithm 2, then map x' offsets → x offsets
8.  reject spans with alignment status "rejected" (kept as flagged records, counted as FP)
9.  return A*
```
Ablation switches: line 1 → generic specification (A1); line 2 → P_fixed (A2, B0); line 3 →
identity (A3, B0); lines 5–6 skipped (A4, B0); R = 0 (A6); Algorithm 2 restricted to exact matching
(A5, B0).  Reflection (R2) re-enters at line 4 with the R1 output appended to the prompt.

## Algorithm 2 – Text alignment (`zengziagent/alignment.py`)

```
Input : source text x', generated span a, optional index hint h, previous span end e_prev,
        similarity threshold θ (0.80)
Output: (status, char span [s, t), token range [i, j))

1.  occ ← all verbatim occurrences of a in x'
    if occ ≠ ∅: s ← tie_break(occ, h, e_prev); return ("exact", [s, s+|a|), tokens([s, s+|a|)))
2.  (x̃, map) ← Normalise(x')   # NFKC, quote/dash unification, whitespace collapse, case fold
    ã ← Normalise(a)
    occ ← occurrences of ã (and of ã stripped of boundary punctuation) in x̃
    if occ ≠ ∅: s ← tie_break(occ, h, e_prev); return ("normalized", map[s..], tokens(...))
3.  B ← matching blocks of SequenceMatcher(x̃, ã) (autojunk off) with size ≥ 4
    for each of the longest ≤ 6 blocks (α, β, n): window ← [α − β, α − β + |ã|)
        refine window boundaries (coarse step 4, fine step 1) to maximise ratio(x̃[window], ã)
    (ρ, s, t) ← best window (ties → tie_break)
    if ρ < θ: return ("rejected", ∅)
    (s, t) ← snap to token boundaries; return ("fuzzy", map[s..t], tokens([s, t)))

tie_break(occ, h, e_prev): occurrence closest to h if a hint is given,
                           else the first occurrence at or after e_prev, else the first occurrence.
```
Token IoU used by the evaluation: `IoU_tok = |T_p ∩ T_g| / |T_p ∪ T_g|` over tokenizer spans of
the original text; `TP = 1[y_p = y_g ∧ IoU_tok ≥ τ]` with τ = 0.5 (sensitivity: 0.3–1.0).
