"""Agreement statistics: Cohen's kappa, Fleiss' kappa, one-way ICC, inter-run agreement."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a = list(a)
    b = list(b)
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(matrix: np.ndarray) -> float:
    """``matrix``: subjects x categories counts (each row sums to the number of raters)."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return float("nan")
    n_raters = m.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("all subjects must have the same number of raters")
    k = n_raters[0]
    p_j = m.sum(axis=0) / m.sum()
    P_i = ((m * m).sum(axis=1) - k) / (k * (k - 1))
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def icc_oneway(ratings: np.ndarray) -> float:
    """ICC(1,1): ``(MSB - MSW) / (MSB + (k-1) MSW)`` for a subjects x raters matrix."""
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    grand = x.mean()
    msb = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def inter_run_agreement(run_labels: Sequence[Sequence]) -> dict:
    """Agreement of predicted unit labels across repeated runs (same units, same order)."""
    runs = [list(r) for r in run_labels]
    if len(runs) < 2:
        return {"pairwise_agreement": float("nan"), "fleiss_kappa": float("nan"), "n_runs": len(runs)}
    n = len(runs[0])
    pair_scores = []
    kappas = []
    for a, b in combinations(runs, 2):
        pair_scores.append(sum(1 for x, y in zip(a, b) if x == y) / n if n else float("nan"))
        kappas.append(cohen_kappa(a, b))
    cats = sorted({str(x) for r in runs for x in r})
    idx = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((n, len(cats)))
    for r in runs:
        for i, x in enumerate(r):
            mat[i, idx[str(x)]] += 1
    return {
        "pairwise_agreement": float(np.mean(pair_scores)),
        "pairwise_agreement_sd": float(np.std(pair_scores, ddof=1)) if len(pair_scores) > 1 else 0.0,
        "pairwise_cohen_kappa": float(np.mean(kappas)),
        "fleiss_kappa": float(fleiss_kappa(mat)) if n else float("nan"),
        "n_runs": len(runs),
        "n_units": n,
    }
