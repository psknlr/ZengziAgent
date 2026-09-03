"""Tokenisers used for token-boundary mapping and token IoU.

spaCy's rule-based English tokenizer (``spacy.blank("en")``) is used when
available, exactly as in the Text Alignment Tool description; a regular-expression
tokenizer with identical span semantics is the fallback so that evaluation never
silently changes because of a missing optional dependency (the manifest records
which tokenizer was used).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Protocol, Sequence

TokenSpans = Sequence[tuple[int, int]]


class Tokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:  # pragma: no cover - protocol
        ...


class _SpanCache:
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._d: "OrderedDict[str, list[tuple[int, int]]]" = OrderedDict()

    def get(self, key: str):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: str, value):
        self._d[key] = value
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


class RegexTokenizer:
    """Word/punctuation tokenizer: ``\\w+`` (with internal hyphens/apostrophes) or single punctuation."""

    name = "regex"
    TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)

    def __init__(self):
        self._cache = _SpanCache()

    def spans(self, text: str) -> list[tuple[int, int]]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        out = [(m.start(), m.end()) for m in self.TOKEN_RE.finditer(text)]
        self._cache.put(text, out)
        return out


class SpacyTokenizer:
    name = "spacy-blank-en"

    def __init__(self):
        import spacy  # noqa: WPS433

        self._nlp = spacy.blank("en")
        self._nlp.max_length = 5_000_000
        self._cache = _SpanCache()

    def spans(self, text: str) -> list[tuple[int, int]]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        doc = self._nlp.make_doc(text)
        out = [(t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]
        self._cache.put(text, out)
        return out


_DEFAULT: Tokenizer | None = None


def get_tokenizer(name: str = "auto") -> Tokenizer:
    """Return a tokenizer; ``auto`` prefers spaCy and falls back to the regex tokenizer."""
    global _DEFAULT
    if name == "auto":
        if _DEFAULT is None:
            try:
                _DEFAULT = SpacyTokenizer()
            except Exception:
                _DEFAULT = RegexTokenizer()
        return _DEFAULT
    if name in {"spacy", "spacy-blank-en"}:
        return SpacyTokenizer()
    if name == "regex":
        return RegexTokenizer()
    raise ValueError(f"unknown tokenizer: {name}")


def char_span_to_token_range(token_spans: TokenSpans, start: int, end: int) -> tuple[int, int]:
    """Map a character span ``[start, end)`` to the token index range ``[i, j)`` of all
    tokens overlapping it.  Returns ``(k, k)`` (empty) when no token overlaps."""
    i = None
    j = None
    for idx, (ts, te) in enumerate(token_spans):
        if te <= start:
            continue
        if ts >= end:
            break
        if i is None:
            i = idx
        j = idx + 1
    if i is None:
        # empty: locate insertion point
        k = 0
        for idx, (ts, _te) in enumerate(token_spans):
            if ts < start:
                k = idx + 1
        return (k, k)
    return (i, j)


def token_set(token_spans: TokenSpans, start: int, end: int) -> set[int]:
    i, j = char_span_to_token_range(token_spans, start, end)
    return set(range(i, j))


def token_iou(token_spans: TokenSpans, a: tuple[int, int], b: tuple[int, int]) -> float:
    """``IoU_tok = |T_p ∩ T_g| / |T_p ∪ T_g|`` for two character spans."""
    ai, aj = char_span_to_token_range(token_spans, a[0], a[1])
    bi, bj = char_span_to_token_range(token_spans, b[0], b[1])
    if aj <= ai or bj <= bi:
        return 0.0
    inter = max(0, min(aj, bj) - max(ai, bi))
    union = (aj - ai) + (bj - bi) - inter
    return inter / union if union else 0.0


def token_overlap(token_spans: TokenSpans, a: tuple[int, int], b: tuple[int, int]) -> int:
    ai, aj = char_span_to_token_range(token_spans, a[0], a[1])
    bi, bj = char_span_to_token_range(token_spans, b[0], b[1])
    return max(0, min(aj, bj) - max(ai, bi))


def snap_to_token_boundaries(token_spans: TokenSpans, start: int, end: int) -> tuple[int, int]:
    """Boundary correction: expand ``[start, end)`` to the boundaries of the tokens it touches."""
    i, j = char_span_to_token_range(token_spans, start, end)
    if j <= i:
        return start, end
    return token_spans[i][0], token_spans[j - 1][1]
