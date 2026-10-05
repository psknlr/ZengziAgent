"""Stage 3 - deterministic text preprocessing (the component formerly called *Summarizer*).

The component never rewrites reviewer wording.  It performs conservative,
*offset-preserving* normalisation so that every character of the processed text
can be mapped back to the original reviewer comment:

* per-character Unicode NFKC normalisation;
* zero-width character removal;
* removal of well-formed non-content markup tags (``<br>``, ``<i>`` ...);
* whitespace standardisation (runs of blanks -> one space, runs of line breaks -> one ``\\n``);
* sentence-boundary detection for unit construction.

When preprocessing is ablated (configuration A3) the identity mapping is used.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]{0,200})?/?>")
_ZERO_WIDTH = {"​", "‌", "‍", "﻿", "⁠", "­"}
_SENT_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[A-Z0-9\"'“(\[])|\n+")


@dataclass
class ProcessedText:
    original: str
    text: str
    to_original: list[int] = field(repr=False)  # processed index -> original index
    sentences: list[tuple[int, int]] = field(default_factory=list)  # processed offsets
    identity: bool = False

    def map_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a processed-text span ``[start, end)`` to original-text offsets."""
        if self.identity or not self.to_original:
            return start, end
        n = len(self.to_original)
        if end <= start or n == 0:
            s = self.to_original[min(max(start, 0), n - 1)]
            return s, s
        s = self.to_original[min(max(start, 0), n - 1)]
        e = self.to_original[min(max(end - 1, 0), n - 1)] + 1
        return s, e

    def map_original_to_processed(self, start: int, end: int) -> tuple[int, int]:
        """Inverse map (used to project gold spans into processed coordinates for demonstrations)."""
        if self.identity or not self.to_original:
            return start, end
        ps = None
        pe = None
        for i, o in enumerate(self.to_original):
            if ps is None and o >= start:
                ps = i
            if o < end:
                pe = i + 1
        if ps is None:
            ps = len(self.to_original)
        if pe is None or pe < ps:
            pe = ps
        return ps, pe


def segment_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence segmentation returning ``[start, end)`` offsets (trimmed of whitespace).

    Boundaries are placed after terminal punctuation followed by whitespace and an
    upper-case/digit/quote start, and at line breaks -- except *soft wraps*, i.e. a line
    break whose preceding line does not end with terminal punctuation and whose next
    line starts with a lower-case letter.
    """
    spans = []
    pos = 0
    for m in _SENT_RE.finditer(text):
        if m.group(0).startswith("\n"):
            before = text[:m.start()].rstrip()
            after = text[m.end():m.end() + 1]
            if before and before[-1] not in ".!?:;\"'”’)]" and after.islower():
                continue  # soft wrap inside a sentence
            seg = (pos, m.start())
        else:
            seg = (pos, m.start() + _trailing_quote_len(text, m.start()))
        pos = m.end()
        spans.append(seg)
    spans.append((pos, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def _trailing_quote_len(text: str, idx: int) -> int:
    n = 0
    while idx + n < len(text) and text[idx + n] in "\"'”’)]":
        n += 1
    return n


class Preprocessor:
    def __init__(self, enabled: bool = True, strip_markup: bool = True, nfkc: bool = True):
        self.enabled = enabled
        self.strip_markup = strip_markup
        self.nfkc = nfkc

    @property
    def name(self) -> str:
        return "deterministic-nfkc-whitespace-markup" if self.enabled else "identity"

    def process(self, text: str) -> ProcessedText:
        if not self.enabled:
            pt = ProcessedText(original=text, text=text, to_original=list(range(len(text))), identity=True)
            pt.sentences = segment_sentences(text)
            return pt

        skip = [False] * len(text)
        if self.strip_markup:
            for m in _TAG_RE.finditer(text):
                for k in range(m.start(), m.end()):
                    skip[k] = True

        out_chars: list[str] = []
        out_map: list[int] = []
        pending_newline = False
        pending_space = False
        ws_idx = 0  # original index of the first whitespace character of the pending run

        def emit(ch: str, idx: int) -> None:
            out_chars.append(ch)
            out_map.append(idx)

        for i, ch in enumerate(text):
            if skip[i] or ch in _ZERO_WIDTH:
                continue
            norm = unicodedata.normalize("NFKC", ch) if self.nfkc else ch
            if not norm:
                continue
            for nc in norm:
                if nc in "\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029":
                    if not (pending_newline or pending_space):
                        ws_idx = i
                    pending_newline = True
                    continue
                if nc.isspace():
                    if not (pending_newline or pending_space):
                        ws_idx = i
                    pending_space = True
                    continue
                if pending_newline:
                    if out_chars:
                        emit("\n", ws_idx)
                elif pending_space:
                    if out_chars:
                        emit(" ", ws_idx)
                pending_newline = False
                pending_space = False
                emit(nc, i)
        processed = "".join(out_chars)
        pt = ProcessedText(original=text, text=processed, to_original=out_map, identity=False)
        pt.sentences = segment_sentences(processed)
        return pt
