#!/usr/bin/env python3
"""Download SubstanReview, (optionally) fetch eLife reviewer text, and write evaluation manifests.

Examples::

    python scripts/prepare_data.py --substanreview
    python scripts/prepare_data.py --elife-ids data/elife/article_ids.txt
    python scripts/prepare_data.py --elife-sample 40 --years 2016 2017 2018 2019 2020 --mailto you@example.org
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zengziagent.data.elife import build_review_records, list_elife_articles_openalex  # noqa: E402
from zengziagent.data.manifest import write_manifest  # noqa: E402
from zengziagent.data.substanreview import download_substanreview, load_substanreview  # noqa: E402
from zengziagent.data.units import save_records_jsonl  # noqa: E402
from zengziagent.utils import setup_logging, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substanreview", action="store_true", help="download SubstanReview train/test/IAA files")
    ap.add_argument("--sr-root", default="data/substanreview")
    ap.add_argument("--elife-ids", help="text file with one eLife article id per line")
    ap.add_argument("--elife-sample", type=int, default=0, help="sample N articles per year via OpenAlex")
    ap.add_argument("--years", type=int, nargs="*", default=[2016, 2017, 2018, 2019, 2020])
    ap.add_argument("--mailto", default=None, help="contact e-mail for the OpenAlex polite pool")
    ap.add_argument("--elife-out", default="data/elife/reviews.jsonl")
    ap.add_argument("--granularity", choices=["article", "reviewer"], default="article")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()

    if args.substanreview:
        paths = download_substanreview(args.sr_root)
        print("downloaded:", {k: str(v) for k, v in paths.items()})
        for split in ("train", "test"):
            recs = load_substanreview(args.sr_root, split)
            m = write_manifest(recs, "substanreview", split, Path(args.sr_root) / f"manifest_{split}.json")
            print(f"substanreview/{split}: {m['n_records']} reviews, {m['statistics']['total_units']} units, hash {m['dataset_hash'][:12]}")

    ids: list[str] = []
    if args.elife_ids:
        ids = [l.strip() for l in open(args.elife_ids, encoding="utf-8") if l.strip() and not l.startswith("#")]
    if args.elife_sample:
        sampled = []
        for y in args.years:
            works = list_elife_articles_openalex(y, args.elife_sample, seed=args.seed + y, mailto=args.mailto)
            print(f"eLife {y}: sampled {len(works)} articles")
            sampled.extend(works)
        write_json(Path(args.elife_out).with_name("sampled_articles.json"), sampled)
        ids.extend(w["article_id"] for w in sampled)
    if ids:
        recs = build_review_records(ids, granularity=args.granularity)
        n = save_records_jsonl(recs, args.elife_out)
        write_manifest(recs, "elife", "extracted", Path(args.elife_out).with_name("manifest_extracted.json"))
        print(f"eLife: wrote {n} reviewer records to {args.elife_out} (gold labels must be attached separately)")


if __name__ == "__main__":
    main()
