"""Annotation units, canonical JSONL I/O and corpus statistics (Table 4)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from ..preprocessor import segment_sentences
from ..schema import LABELS, AnnotationUnit, GoldSpan, ReviewRecord, canonical_label
from ..tokenization import get_tokenizer, token_overlap
from ..utils import LOG, read_jsonl, sha256_text, write_jsonl


def load_records_jsonl(path: str | Path, dataset: str | None = None) -> list[ReviewRecord]:
    """Load reviews from JSONL.  Two layouts are accepted:

    * canonical: ``{"review_id", "dataset", "text", "gold_spans": [{start,end,label}], "metadata"}``
    * SubstanReview style: ``{"id", "review", "label": [[start, end, "Eval_pos_1"], ...], "meta"?}``
    """
    out: list[ReviewRecord] = []
    for row in read_jsonl(path):
        if "text" in row and "review_id" in row and "gold_spans" in row:
            rec = ReviewRecord.from_dict(row)
            if dataset:
                rec.dataset = dataset
            spans = []
            for g in rec.gold_spans:  # canonical layout is validated exactly like the release layout
                lab, occ = canonical_label(g.label)
                if lab is None:
                    LOG.warning("skipping unknown gold label %r in review %s", g.label, rec.review_id)
                    continue
                spans.append(GoldSpan(start=g.start, end=g.end, label=lab, occurrence=g.occurrence if g.occurrence is not None else occ, raw_label=g.raw_label or g.label))
            rec.gold_spans = spans
        else:
            spans = []
            for item in row.get("label", []) or []:
                s, e, raw = int(item[0]), int(item[1]), str(item[2])
                lab, occ = canonical_label(raw)
                if lab is None:
                    LOG.warning("skipping unknown gold label %r in review %s", raw, row.get("id"))
                    continue
                spans.append(GoldSpan(start=s, end=e, label=lab, occurrence=occ, raw_label=raw))
            text = row["review"] if "review" in row else row["text"]  # IAA files use "text"
            meta = dict(row.get("meta", row.get("metadata", {})) or {})
            for extra in ("rid", "scores", "Comments"):
                if extra in row:
                    meta[extra] = row[extra]
            rec = ReviewRecord(
                review_id=str(row.get("review_id", row.get("id"))),
                dataset=dataset or row.get("dataset", "unknown"),
                text=text,
                gold_spans=spans,
                metadata=meta,
            )
        rec.gold_spans = _trim_boundary_whitespace(rec.text, rec.gold_spans, rec.review_id)
        rec.gold_spans.sort(key=lambda g: (g.start, g.end))
        out.append(rec)
    return out


def _trim_boundary_whitespace(text: str, spans: list[GoldSpan], review_id: str) -> list[GoldSpan]:
    """Gold offsets in the SubstanReview release frequently include a leading/trailing space
    (23% of spans).  Model spans are stripped before alignment, so boundary whitespace would
    make exact span matches unattainable; offsets are trimmed here and the raw ones kept."""
    out = []
    for g in spans:
        s, e = g.start, g.end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s:
            LOG.warning("dropping empty gold span %s-%s in review %s", g.start, g.end, review_id)
            continue
        if (s, e) != (g.start, g.end):
            g.raw_label = g.raw_label or g.label
            g = GoldSpan(start=s, end=e, label=g.label, occurrence=g.occurrence, raw_label=g.raw_label)
        out.append(g)
    return out


def save_records_jsonl(records: Iterable[ReviewRecord], path: str | Path) -> int:
    return write_jsonl(path, (r.to_dict() for r in records))


def build_units(review: ReviewRecord, mode: str = "gold_span", tokenizer=None) -> list[AnnotationUnit]:
    """Construct evaluated annotation units.

    ``gold_span``: one unit per gold span (the setting of Table 4: 4,160 SubstanReview units).
    ``sentence``: one unit per sentence, labelled with the gold span of largest token
    overlap (``None`` when no gold span overlaps) - used to train unit-classification baselines.
    """
    units: list[AnnotationUnit] = []
    if mode == "gold_span":
        for k, g in enumerate(review.gold_spans):
            units.append(
                AnnotationUnit(
                    unit_id=f"{review.review_id}#{k}",
                    review_id=review.review_id,
                    dataset=review.dataset,
                    start=g.start,
                    end=g.end,
                    gold_label=g.label,
                    text=review.text[g.start : g.end],
                )
            )
        return units
    if mode == "sentence":
        tok = tokenizer or get_tokenizer("auto")
        tspans = tok.spans(review.text)
        for k, (s, e) in enumerate(segment_sentences(review.text)):
            best, best_ov = "None", 0
            for g in review.gold_spans:
                ov = token_overlap(tspans, (s, e), (g.start, g.end))
                if ov > best_ov:
                    best, best_ov = g.label, ov
            units.append(AnnotationUnit(f"{review.review_id}#s{k}", review.review_id, review.dataset, s, e, best, review.text[s:e]))
        return units
    raise ValueError(f"unknown unit mode {mode}")


def corpus_statistics(records: list[ReviewRecord], tokenizer=None) -> dict:
    """Table 4 style statistics under the mapped five-label scheme."""
    tok = tokenizer or get_tokenizer("auto")
    n_reviews = len(records)
    label_counts: Counter = Counter()
    n_units = 0
    unit_tokens = 0
    starts: list[int] = []
    ends: list[int] = []
    for r in records:
        tspans = tok.spans(r.text)
        for g in r.gold_spans:
            label_counts[g.label] += 1
            n_units += 1
            unit_tokens += len([1 for ts, te in tspans if te > g.start and ts < g.end])
            starts.append(g.start)
            ends.append(g.end)
    return {
        "total_reviews": n_reviews,
        "total_units": n_units,
        "avg_units_per_review": round(n_units / n_reviews, 2) if n_reviews else 0.0,
        "avg_tokens_per_unit": round(unit_tokens / n_units, 2) if n_units else 0.0,
        "label_counts": {lab: label_counts.get(lab, 0) for lab in LABELS},
        "label_shares": {lab: round(label_counts.get(lab, 0) / n_units, 4) if n_units else 0.0 for lab in LABELS},
        "start_index_range": [min(starts), max(starts)] if starts else None,
        "end_index_range": [min(ends), max(ends)] if ends else None,
        "tokenizer": getattr(tok, "name", "unknown"),
    }


def dataset_hash(records: list[ReviewRecord]) -> str:
    """Content hash over (review_id, text, gold spans) - stored in every run manifest."""
    parts = []
    for r in sorted(records, key=lambda x: x.review_id):
        spans = ";".join(f"{g.start}-{g.end}-{g.label}" for g in r.gold_spans)
        parts.append(f"{r.review_id}\t{sha256_text(r.text)}\t{spans}")
    return sha256_text("\n".join(parts))
