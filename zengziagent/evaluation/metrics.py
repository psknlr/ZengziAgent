"""Unit-level accuracy, span-level micro P/R/F1 with a token-IoU criterion, and alignment metrics.

Definitions (Section 4.3.3 of the revised manuscript):

* unit-level accuracy  ``Acc = N_correct / N`` over evaluated annotation units, where the
  predicted label of a unit is the label of the recovered predicted span with the largest
  token overlap with the unit (``None`` when no predicted span overlaps it);
* span-level ``TP = 1[y_p = y_g and IoU_tok >= tau]`` under one-to-one greedy matching
  (highest IoU first, label-aware), ``P = TP/(TP+FP)``, ``R = TP/(TP+FN)``,
  ``F1 = 2PR/(P+R)``; micro-averaged over labels; *pooled across datasets* means summing
  the counts, never averaging the ratios;
* alignment metrics: exact span match rate (gold spans reproduced with identical boundaries
  and label), mean token IoU (per gold span: best IoU with any recovered predicted span),
  recoverable span rate, rejected span rate.

Rejected spans (unrecoverable from the source) are predictions that do not exist in the
text; the default policy counts them as false positives.  Illegal-label spans (only present
when validation is ablated) are always false positives.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..schema import LABELS, AnnotationUnit, ReviewRecord
from ..tokenization import get_tokenizer, token_iou, token_overlap
from .. import data as _data


@dataclass
class PredSpan:
    start: Optional[int]
    end: Optional[int]
    label: Optional[str]
    status: str = "exact"  # exact | normalized | fuzzy | rejected | illegal_label
    similarity: float = 1.0
    text: str = ""

    @property
    def recovered(self) -> bool:
        return self.status in {"exact", "normalized", "fuzzy"} and self.start is not None and self.end is not None and self.end > self.start


@dataclass
class ReviewEvaluation:
    review_id: str
    dataset: str
    n_units: int = 0
    n_correct: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred: int = 0
    n_exact: int = 0
    n_normalized: int = 0
    n_fuzzy: int = 0
    n_rejected: int = 0
    n_illegal: int = 0
    exact_span_matches: int = 0
    sum_best_iou: float = 0.0
    per_label: dict = field(default_factory=dict)  # label -> {units, correct, tp, fp, fn, n_pred}
    unit_records: list[dict] = field(default_factory=list)
    confusion: dict = field(default_factory=dict)  # (gold, pred) -> count

    def counts(self) -> list:
        return [self.n_units, self.n_correct, self.tp, self.fp, self.fn]

    def to_row(self) -> dict:
        row = {
            "review_id": self.review_id,
            "dataset": self.dataset,
            "n_units": self.n_units,
            "n_correct": self.n_correct,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_pred": self.n_pred,
            "n_exact": self.n_exact,
            "n_normalized": self.n_normalized,
            "n_fuzzy": self.n_fuzzy,
            "n_rejected": self.n_rejected,
            "n_illegal": self.n_illegal,
            "exact_span_matches": self.exact_span_matches,
            "sum_best_iou": round(self.sum_best_iou, 6),
        }
        for lab in LABELS:
            pl = self.per_label.get(lab, {})
            for k in ("units", "correct", "tp", "fp", "fn", "n_pred"):
                row[f"{lab}__{k}"] = pl.get(k, 0)
        return row


def pred_spans_from_result(result: dict) -> list[PredSpan]:
    """Build predicted spans from a serialised ``ReviewResult`` (original-text coordinates)."""
    out = []
    for s in result.get("spans", []):
        status = s.get("status", "rejected")
        label = s.get("label")
        if status == "illegal_label" or (label not in LABELS and status in {"exact", "normalized", "fuzzy"}):
            status = "illegal_label"
        out.append(PredSpan(s.get("original_start"), s.get("original_end"), label, status, float(s.get("similarity", 0.0) or 0.0), s.get("text", "")))
    return out


def evaluate_review(
    review: ReviewRecord,
    preds: list[PredSpan],
    tau: float = 0.5,
    tokenizer=None,
    rejected_policy: str = "count_as_fp",
    units: Optional[list[AnnotationUnit]] = None,
) -> ReviewEvaluation:
    tok = tokenizer or get_tokenizer("auto")
    tspans = tok.spans(review.text)
    ev = ReviewEvaluation(review_id=review.review_id, dataset=review.dataset)
    units = units if units is not None else _data.build_units(review, "gold_span")
    recovered = [p for p in preds if p.recovered]
    ev.n_pred = len(preds)
    ev.n_exact = sum(1 for p in preds if p.status == "exact")
    ev.n_normalized = sum(1 for p in preds if p.status == "normalized")
    ev.n_fuzzy = sum(1 for p in preds if p.status == "fuzzy")
    ev.n_rejected = sum(1 for p in preds if p.status == "rejected")
    ev.n_illegal = sum(1 for p in preds if p.status == "illegal_label")

    per_label: dict[str, dict] = {lab: {"units": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0, "n_pred": 0} for lab in LABELS}
    for p in recovered:
        if p.label in per_label:
            per_label[p.label]["n_pred"] += 1

    # ---- span-level matching (label-aware, greedy by IoU, one-to-one)
    gold = review.gold_spans
    pairs = []
    for i, p in enumerate(recovered):
        if p.label not in per_label:
            continue
        for j, g in enumerate(gold):
            if g.label != p.label:
                continue
            iou = token_iou(tspans, (p.start, p.end), (g.start, g.end))
            if iou >= tau and iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        per_label[gold[j].label]["tp"] += 1
    n_gold_by_label = Counter(g.label for g in gold)
    for lab in LABELS:
        per_label[lab]["fp"] = per_label[lab]["n_pred"] - per_label[lab]["tp"]
        per_label[lab]["fn"] = n_gold_by_label.get(lab, 0) - per_label[lab]["tp"]
    ev.tp = sum(v["tp"] for v in per_label.values())
    ev.fp = sum(v["fp"] for v in per_label.values()) + ev.n_illegal + (ev.n_rejected if rejected_policy == "count_as_fp" else 0)
    ev.fn = sum(v["fn"] for v in per_label.values())

    # ---- unit-level accuracy + alignment fidelity per gold unit
    confusion: Counter = Counter()
    for u in units:
        best_lab, best_ov, best_iou_same = None, 0, 0.0
        best_iou_any = 0.0
        exact = False
        for p in recovered:
            ov = token_overlap(tspans, (p.start, p.end), (u.start, u.end))
            iou = token_iou(tspans, (p.start, p.end), (u.start, u.end))
            best_iou_any = max(best_iou_any, iou)
            if p.label == u.gold_label and p.start == u.start and p.end == u.end:
                exact = True
            if ov > best_ov or (ov == best_ov and ov > 0 and iou > best_iou_same):
                best_lab, best_ov, best_iou_same = p.label, ov, iou
        pred_label = best_lab if best_ov > 0 else None
        correct = pred_label == u.gold_label
        ev.n_units += 1
        ev.n_correct += int(correct)
        ev.sum_best_iou += best_iou_any
        ev.exact_span_matches += int(exact)
        if u.gold_label in per_label:
            per_label[u.gold_label]["units"] += 1
            per_label[u.gold_label]["correct"] += int(correct)
        confusion[(u.gold_label, pred_label or "None")] += 1
        ev.unit_records.append({"unit_id": u.unit_id, "review_id": review.review_id, "dataset": review.dataset, "gold_label": u.gold_label, "pred_label": pred_label, "correct": int(correct), "best_iou": round(best_iou_any, 4), "exact": int(exact)})
    ev.per_label = per_label
    ev.confusion = {f"{g}->{p}": c for (g, p), c in confusion.items()}
    return ev


def metrics_from_counts(n_units: int, n_correct: int, tp: int, fp: int, fn: int) -> dict:
    acc = n_correct / n_units if n_units else float("nan")
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def aggregate(evals: list[ReviewEvaluation]) -> dict:
    """Pool counts across reviews (and across datasets when asked) and derive the metrics."""
    tot = {"n_reviews": len(evals), "n_units": 0, "n_correct": 0, "tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_exact": 0, "n_normalized": 0, "n_fuzzy": 0, "n_rejected": 0, "n_illegal": 0, "exact_span_matches": 0, "sum_best_iou": 0.0}
    per_label = {lab: {"units": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0, "n_pred": 0} for lab in LABELS}
    confusion: Counter = Counter()
    for e in evals:
        for k in list(tot):
            if k == "n_reviews":
                continue
            tot[k] += getattr(e, k)
        for lab in LABELS:
            for k in per_label[lab]:
                per_label[lab][k] += e.per_label.get(lab, {}).get(k, 0)
        confusion.update(e.confusion)
    out = dict(tot)
    out.update(metrics_from_counts(tot["n_units"], tot["n_correct"], tot["tp"], tot["fp"], tot["fn"]))
    n_pred = tot["n_pred"]
    n_recovered = tot["n_exact"] + tot["n_normalized"] + tot["n_fuzzy"]
    out["recoverable_span_rate"] = n_recovered / n_pred if n_pred else float("nan")
    out["rejected_span_rate"] = tot["n_rejected"] / n_pred if n_pred else float("nan")
    out["fuzzy_span_rate"] = tot["n_fuzzy"] / n_pred if n_pred else float("nan")
    out["exact_span_match_rate"] = tot["exact_span_matches"] / tot["n_units"] if tot["n_units"] else float("nan")
    out["mean_token_iou"] = tot["sum_best_iou"] / tot["n_units"] if tot["n_units"] else float("nan")
    out["per_label"] = {}
    for lab in LABELS:
        pl = per_label[lab]
        m = metrics_from_counts(pl["units"], pl["correct"], pl["tp"], pl["fp"], pl["fn"])
        out["per_label"][lab] = {**pl, **m}
    out["confusion"] = dict(confusion)
    return out
