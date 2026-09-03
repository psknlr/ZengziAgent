"""Evaluation manifests: exact record ids, hashes and label counts for each split."""
from __future__ import annotations

from pathlib import Path

from ..schema import ReviewRecord
from ..utils import sha256_text, utc_now, write_json
from .units import corpus_statistics, dataset_hash


def build_manifest(records: list[ReviewRecord], dataset: str, split: str, extra: dict | None = None) -> dict:
    stats = corpus_statistics(records)
    return {
        "dataset": dataset,
        "split": split,
        "created_at": utc_now(),
        "n_records": len(records),
        "dataset_hash": dataset_hash(records),
        "statistics": stats,
        "records": [
            {"review_id": r.review_id, "text_sha256": sha256_text(r.text), "n_chars": len(r.text), "n_gold_spans": len(r.gold_spans), "metadata": r.metadata}
            for r in records
        ],
        **(extra or {}),
    }


def write_manifest(records: list[ReviewRecord], dataset: str, split: str, path: str | Path, extra: dict | None = None) -> dict:
    m = build_manifest(records, dataset, split, extra)
    write_json(path, m)
    return m
