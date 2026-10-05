"""Build the article-level feature table: predictions + article metadata + OpenAlex citations.

Example::

    python -m zengziagent.scientometrics.build_features --predictions results/raw/elife/openrouter-claude/F/run0/predictions.jsonl \
        --records data/elife/gold/elife_gold.jsonl --mailto you@example.org --out results/scientometrics/article_features.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.units import load_records_jsonl
from ..utils import LOG, setup_logging
from .features import article_features
from .openalex import fetch_citations


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--records", required=True, help="eLife records JSONL with metadata (article_id, year, doi)")
    ap.add_argument("--citations-cache", default="data/elife/citations_cache.json")
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out", default="results/scientometrics/article_features.csv")
    args = ap.parse_args(argv)
    setup_logging()
    recs = load_records_jsonl(args.records, dataset="elife")
    meta = {r.review_id: r.metadata for r in recs}
    df = article_features(args.predictions, meta)
    dois = [d for d in df["doi"].dropna().unique().tolist() if d]
    years = {row.doi: int(row.year) for row in df.itertuples() if row.doi and row.year == row.year}
    cache = fetch_citations(dois, cache_path=args.citations_cache, mailto=args.mailto, years=years, window=args.window)
    df["citations"] = [cache.get(str(d).lower(), {}).get("citations_5y") for d in df["doi"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOG.info("wrote %s (%d articles, %d with citations)", args.out, len(df), int(df["citations"].notna().sum()))


if __name__ == "__main__":  # pragma: no cover
    main()
