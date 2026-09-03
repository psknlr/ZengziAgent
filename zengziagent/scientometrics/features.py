"""Annotation-derived review features (Section 4.3.4).

Let ``N_Eval+, N_Eval-, N_Jus+, N_Jus-, N_MC`` be the per-review span counts,
``N_Eval = N_Eval+ + N_Eval-``, ``N_Jus = N_Jus+ + N_Jus-``, ``N = N_Eval + N_Jus + N_MC``.

core set (4):      r_substan = N_Jus / (N_Eval + N_Jus); r_neg_val = N_Eval- / N_Eval;
                   r_jus = N_Jus / N; T = N
extended set (6):  + r_crit = N_Eval- / N; r_neg_jus = N_Jus- / N_Jus
full set (8):      + b_pos = N_Eval+ / (N_Eval- + 1); c_MC = N_MC
Reviews are aggregated to the article level (sum of counts) *before* ratios are formed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ..schema import LABELS
from ..utils import read_jsonl

CORE = ["r_substan", "r_neg_val", "r_jus", "T"]
EXTENDED = CORE + ["r_crit", "r_neg_jus"]
FULL = EXTENDED + ["b_pos", "c_MC"]


def span_counts(spans: list[dict], recovered_only: bool = True) -> Counter:
    c: Counter = Counter()
    for s in spans:
        if recovered_only and s.get("status") not in {"exact", "normalized", "fuzzy"}:
            continue
        if s.get("label") in LABELS:
            c[s["label"]] += 1
    return c


def features_from_counts(c: Counter) -> dict:
    n_ep, n_en, n_jp, n_jn, n_mc = (c.get(l, 0) for l in LABELS)
    n_eval = n_ep + n_en
    n_jus = n_jp + n_jn
    n = n_eval + n_jus + n_mc

    def div(a, b):
        return a / b if b else float("nan")

    return {
        "N_Eval_pos": n_ep,
        "N_Eval_neg": n_en,
        "N_Jus_pos": n_jp,
        "N_Jus_neg": n_jn,
        "N_MC": n_mc,
        "N": n,
        "r_substan": div(n_jus, n_eval + n_jus),
        "r_neg_val": div(n_en, n_eval),
        "r_jus": div(n_jus, n),
        "T": n,
        "r_crit": div(n_en, n),
        "r_neg_jus": div(n_jn, n_jus),
        "b_pos": n_ep / (n_en + 1),
        "c_MC": n_mc,
    }


def article_features(predictions_jsonl: str | Path, records_meta: dict[str, dict]) -> pd.DataFrame:
    """Aggregate review-level span counts to articles.  ``records_meta`` maps ``review_id`` to
    metadata containing at least ``article_id``, ``year`` and ``doi``."""
    per_article: dict[str, Counter] = {}
    meta: dict[str, dict] = {}
    for row in read_jsonl(predictions_jsonl):
        m = records_meta.get(row["review_id"])
        if m is None:
            continue
        aid = str(m.get("article_id", row["review_id"]))
        per_article.setdefault(aid, Counter()).update(span_counts(row.get("spans", [])))
        meta[aid] = {"article_id": aid, "year": m.get("year"), "doi": m.get("doi")}
    rows = []
    for aid, c in per_article.items():
        rows.append({**meta[aid], **features_from_counts(c)})
    return pd.DataFrame(rows)
