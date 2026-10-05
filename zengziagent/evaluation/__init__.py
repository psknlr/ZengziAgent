"""Single source of truth for all reported numbers.

raw predictions -> :mod:`metrics` (counts per review) -> master CSV -> tables/figures.
"""
from .metrics import PredSpan, ReviewEvaluation, aggregate, evaluate_review, metrics_from_counts, pred_spans_from_result  # noqa: F401
