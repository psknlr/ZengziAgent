"""Inferential statistics for the component-contribution analysis.

* :func:`paired_bootstrap` - paired bootstrap over reviews (the natural resampling unit;
  spans inside a review are not independent) for the difference in micro-F1 or unit
  accuracy between two systems evaluated on the same test cases; 10,000 resamples by
  default; percentile 95% CI; two-sided p-value from the bootstrap distribution.
* :func:`mcnemar` - exact McNemar test on paired unit-level correctness.
* :func:`holm` - Holm step-down adjustment for a family of tests.
* :func:`bootstrap_contrast` - CI for ``ΔF1(eLife) - ΔF1(SubstanReview)``.
* :func:`anova_oneway` - one-way ANOVA F-statistic (per-label accuracy across backends).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _metric_from_matrix(m: np.ndarray, metric: str) -> float:
    """``m`` columns: n_units, n_correct, tp, fp, fn (summed over resampled reviews)."""
    n, correct, tp, fp, fn = m
    if metric == "accuracy":
        return correct / n if n else np.nan
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if metric == "precision":
        return p
    if metric == "recall":
        return r
    return 2 * p * r / (p + r) if (p + r) else 0.0


def paired_bootstrap(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    metric: str = "f1",
    n_resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap for ``metric(A) - metric(B)`` with rows = reviews, cols = counts."""
    a = np.asarray(counts_a, dtype=float)
    b = np.asarray(counts_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 5, "count matrices must be R x 5"
    R = a.shape[0]
    point_a = _metric_from_matrix(a.sum(axis=0), metric)
    point_b = _metric_from_matrix(b.sum(axis=0), metric)
    delta = point_a - point_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(n_resamples, R))
    sa = a[idx].sum(axis=1)  # (B, 5)
    sb = b[idx].sum(axis=1)
    deltas = np.array([_metric_from_matrix(sa[k], metric) - _metric_from_matrix(sb[k], metric) for k in range(n_resamples)])
    deltas = deltas[~np.isnan(deltas)]
    alpha = 1 - confidence
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value (proportion of the bootstrap distribution on the other side of 0)
    p_low = (np.sum(deltas <= 0) + 1) / (len(deltas) + 1)
    p_high = (np.sum(deltas >= 0) + 1) / (len(deltas) + 1)
    p = float(min(1.0, 2 * min(p_low, p_high)))
    return {
        "metric": metric,
        "value_a": float(point_a),
        "value_b": float(point_b),
        "delta": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
        "n_resamples": int(len(deltas)),
        "n_reviews": int(R),
        "boot_mean": float(deltas.mean()),
        "boot_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan"),
    }


def mcnemar(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict:
    """Exact (binomial) McNemar test on paired binary correctness vectors."""
    from scipy import stats

    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    assert a.shape == b.shape
    n_b = int(np.sum(a & ~b))  # correct under A only
    n_c = int(np.sum(~a & b))  # correct under B only
    n = n_b + n_c
    if n == 0:
        return {"b": n_b, "c": n_c, "n_discordant": 0, "p_exact": 1.0, "p_midp": 1.0, "chi2_cc": 0.0, "p_chi2_cc": 1.0}
    k = min(n_b, n_c)
    p_exact = float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))
    p_mid = float(min(1.0, 2 * (stats.binom.cdf(k, n, 0.5) - 0.5 * stats.binom.pmf(k, n, 0.5))))
    chi2 = (abs(n_b - n_c) - 1) ** 2 / n
    p_chi2 = float(stats.chi2.sf(chi2, df=1))
    return {"b": n_b, "c": n_c, "n_discordant": n, "p_exact": p_exact, "p_midp": p_mid, "chi2_cc": float(chi2), "p_chi2_cc": p_chi2}


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def bootstrap_contrast(
    a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray, metric: str = "f1", n_resamples: int = 10000, seed: int = 0, confidence: float = 0.95
) -> dict:
    """CI for ``(metric(A1)-metric(B1)) - (metric(A2)-metric(B2))`` with independent
    review-level resampling inside each dataset (1 and 2)."""
    rng = np.random.default_rng(seed)
    a1, b1, a2, b2 = (np.asarray(x, dtype=float) for x in (a1, b1, a2, b2))
    d1 = _metric_from_matrix(a1.sum(0), metric) - _metric_from_matrix(b1.sum(0), metric)
    d2 = _metric_from_matrix(a2.sum(0), metric) - _metric_from_matrix(b2.sum(0), metric)
    R1, R2 = a1.shape[0], a2.shape[0]
    out = np.empty(n_resamples)
    for k in range(n_resamples):
        i1 = rng.integers(0, R1, size=R1)
        i2 = rng.integers(0, R2, size=R2)
        dd1 = _metric_from_matrix(a1[i1].sum(0), metric) - _metric_from_matrix(b1[i1].sum(0), metric)
        dd2 = _metric_from_matrix(a2[i2].sum(0), metric) - _metric_from_matrix(b2[i2].sum(0), metric)
        out[k] = dd1 - dd2
    alpha = 1 - confidence
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_low = (np.sum(out <= 0) + 1) / (n_resamples + 1)
    p_high = (np.sum(out >= 0) + 1) / (n_resamples + 1)
    return {"delta_1": float(d1), "delta_2": float(d2), "contrast": float(d1 - d2), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, 2 * min(p_low, p_high)))}


def anova_oneway(groups: Sequence[Sequence[float]]) -> dict:
    from scipy import stats

    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"F": float("nan"), "p": float("nan")}
    F, p = stats.f_oneway(*groups)
    return {"F": float(F), "p": float(p)}


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
