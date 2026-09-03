"""Text Alignment Tool (Algorithm 2): source-span recovery and token-boundary mapping.

Given the source text ``x`` and a model-generated span ``a`` the tool returns
``a* = AlignToSource(a, x)``:

1. **Exact match** - verbatim substring search (``str.find``).  When the span occurs
   more than once, the occurrence closest to the model-provided index hint is chosen;
   without a hint, the first occurrence after the previously aligned span is chosen
   (documented tie-breaking rule).
2. **Normalised match** - the same search after conservative normalisation (NFKC,
   quote/dash unification, whitespace collapsing, case folding) with an index map back
   to the source.
3. **Fuzzy recovery** - ``difflib.SequenceMatcher`` (no auto-junk) anchors candidate
   windows on the longest common blocks; each window is scored by the character-level
   similarity ratio and its boundaries are refined and then snapped to token boundaries.
   A candidate is accepted only if ``ratio >= similarity_threshold`` (default 0.80).
4. **Token mapping** - character offsets are converted to token indices with the
   tokenizer of :mod:`zengziagent.tokenization`.

Spans below the threshold are *rejected* (flagged), never silently accepted.  With the
tool disabled (ablation A5) only step 1 is executed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from .tokenization import Tokenizer, char_span_to_token_range, get_tokenizer, snap_to_token_boundaries

_QUOTE_MAP = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    "…": "...",
}


def normalize_for_match(text: str) -> tuple[str, list[int]]:
    """Return ``(normalised_text, index_map)`` where ``index_map[i]`` is the source index of
    normalised character ``i``.  Whitespace runs collapse to one space; text is case-folded."""
    out: list[str] = []
    idx: list[int] = []
    pending_space = False
    ws_idx = 0
    for i, ch in enumerate(text):
        ch = _QUOTE_MAP.get(ch, ch)
        ch = unicodedata.normalize("NFKC", ch)
        for c in ch:
            if c.isspace():
                if not pending_space:
                    ws_idx = i
                pending_space = True
                continue
            if pending_space and out:
                out.append(" ")
                idx.append(ws_idx)
            pending_space = False
            out.append(c.lower())
            idx.append(i)
    return "".join(out), idx


_STRIP_RE = re.compile(r"^[\s\"'“”‘’(\[\-–—.,;:]+|[\s\"'“”‘’)\]\-–—]+$")


@dataclass
class AlignmentResult:
    status: str  # "exact" | "normalized" | "fuzzy" | "rejected"
    start: Optional[int] = None
    end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    similarity: float = 0.0
    method: str = "none"
    n_candidates: int = 0
    recovered_text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return self.status != "rejected"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "start": self.start,
            "end": self.end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "similarity": round(self.similarity, 4),
            "method": self.method,
            "n_candidates": self.n_candidates,
            "recovered_text": self.recovered_text,
        }


class TextAligner:
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        similarity_threshold: float = 0.80,
        allow_normalized: bool = True,
        allow_fuzzy: bool = True,
        boundary_snap: bool = True,
        min_anchor: int = 4,
        max_candidates: int = 6,
        refine_radius: int = 12,
    ):
        self.tokenizer = tokenizer or get_tokenizer("auto")
        self.similarity_threshold = similarity_threshold
        self.allow_normalized = allow_normalized
        self.allow_fuzzy = allow_fuzzy
        self.boundary_snap = boundary_snap
        self.min_anchor = min_anchor
        self.max_candidates = max_candidates
        self.refine_radius = refine_radius
        self._norm_cache: dict[str, tuple[str, list[int]]] = {}

    # ------------------------------------------------------------------ public
    def describe(self) -> dict:
        return {
            "tokenizer": getattr(self.tokenizer, "name", "unknown"),
            "similarity_threshold": self.similarity_threshold,
            "allow_normalized": self.allow_normalized,
            "allow_fuzzy": self.allow_fuzzy,
            "boundary_snap": self.boundary_snap,
            "min_anchor": self.min_anchor,
            "max_candidates": self.max_candidates,
            "refine_radius": self.refine_radius,
            "tie_breaking": "closest to index hint, else first occurrence at/after previous span end",
        }

    def align(
        self,
        source: str,
        span_text: str,
        hint_pos: Optional[int] = None,
        prev_end: int = 0,
    ) -> AlignmentResult:
        span_text = span_text.strip()
        if not span_text:
            return AlignmentResult(status="rejected", method="empty")

        # Step 1: exact (verbatim) match
        occ = _find_all(source, span_text)
        if occ:
            s = _choose_occurrence(occ, len(span_text), hint_pos, prev_end)
            return self._finalize(source, s, s + len(span_text), 1.0, "exact", "exact", len(occ))

        if self.allow_normalized:
            res = self._normalized_match(source, span_text, hint_pos, prev_end)
            if res is not None:
                return res

        if self.allow_fuzzy:
            res = self._fuzzy_match(source, span_text, hint_pos, prev_end)
            if res is not None:
                return res

        return AlignmentResult(status="rejected", method="none", similarity=0.0)

    # ------------------------------------------------------------- internals
    def _source_norm(self, source: str) -> tuple[str, list[int]]:
        # keyed by value (not id()): a str key keeps the source alive and compares by content,
        # so a recycled object id can never serve a stale normalisation of another review
        cached = self._norm_cache.get(source)
        if cached is None:
            cached = normalize_for_match(source)
            if len(self._norm_cache) > 64:
                self._norm_cache.clear()
            self._norm_cache[source] = cached
        return cached

    def _normalized_match(self, source, span_text, hint_pos, prev_end) -> Optional[AlignmentResult]:
        src_norm, src_map = self._source_norm(source)
        span_norm, _ = normalize_for_match(span_text)
        variants = [span_norm]
        stripped = _STRIP_RE.sub("", span_norm).strip()
        if stripped and stripped != span_norm:
            variants.append(stripped)
        for variant in variants:
            if not variant:
                continue
            occ = _find_all(src_norm, variant)
            if occ:
                # map hint/prev_end from source coordinates into normalised coordinates
                hint_n = _map_source_pos_to_norm(src_map, hint_pos) if hint_pos is not None else None
                prev_n = _map_source_pos_to_norm(src_map, prev_end) if prev_end else 0
                s = _choose_occurrence(occ, len(variant), hint_n, prev_n)
                os_, oe = src_map[s], src_map[s + len(variant) - 1] + 1
                return self._finalize(source, os_, oe, 1.0, "normalized", "normalized", len(occ))
        return None

    def _fuzzy_match(self, source, span_text, hint_pos, prev_end) -> Optional[AlignmentResult]:
        src_norm, src_map = self._source_norm(source)
        span_norm, _ = normalize_for_match(span_text)
        if not span_norm or not src_norm:
            return None
        L = len(span_norm)
        sm = SequenceMatcher(None, src_norm, span_norm, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= min(self.min_anchor, L)]
        if not blocks:
            return None
        blocks.sort(key=lambda b: -b.size)
        candidates: list[tuple[float, int, int]] = []
        seen: set[int] = set()
        for b in blocks[: self.max_candidates]:
            cs = max(0, b.a - b.b)
            if cs in seen:
                continue
            seen.add(cs)
            ce = min(len(src_norm), cs + L)
            best = self._refine_window(src_norm, span_norm, cs, ce)
            candidates.append(best)
        if not candidates:
            return None
        # pick the best ratio; tie-break by hint / previous end
        candidates.sort(key=lambda c: -c[0])
        top = candidates[0][0]
        tied = [c for c in candidates if abs(c[0] - top) < 1e-9]
        if len(tied) > 1:
            hint_n = _map_source_pos_to_norm(src_map, hint_pos) if hint_pos is not None else None
            prev_n = _map_source_pos_to_norm(src_map, prev_end) if prev_end else 0
            starts = [c[1] for c in tied]
            s = _choose_occurrence(starts, L, hint_n, prev_n)
            chosen = next(c for c in tied if c[1] == s)
        else:
            chosen = tied[0]
        ratio, cs, ce = chosen
        if ratio < self.similarity_threshold or ce <= cs:
            return AlignmentResult(status="rejected", method="fuzzy_below_threshold", similarity=ratio, n_candidates=len(candidates))
        os_, oe = src_map[cs], src_map[ce - 1] + 1
        return self._finalize(source, os_, oe, ratio, "fuzzy", "fuzzy_sequence_matcher", len(candidates))

    def _refine_window(self, src_norm: str, span_norm: str, cs: int, ce: int) -> tuple[float, int, int]:
        """Coarse-to-fine boundary refinement maximising the similarity ratio."""
        best = (_ratio(src_norm[cs:ce], span_norm), cs, ce)
        n = len(src_norm)
        for step in (4, 1):
            improved = True
            while improved:
                improved = False
                _, bs, be = best
                for ds in (-step, 0, step):
                    for de in (-step, 0, step):
                        if ds == 0 and de == 0:
                            continue
                        s = min(max(0, bs + ds), n)
                        e = min(max(0, be + de), n)
                        if e - s < 1 or abs((e - s) - len(span_norm)) > self.refine_radius * 2 + len(span_norm) // 2:
                            continue
                        r = _ratio(src_norm[s:e], span_norm)
                        if r > best[0] + 1e-9:
                            best = (r, s, e)
                            improved = True
        return best

    def _finalize(self, source, start, end, sim, status, method, n_cand) -> AlignmentResult:
        tok = self.tokenizer.spans(source)
        if self.boundary_snap and status == "fuzzy":
            start, end = snap_to_token_boundaries(tok, start, end)
        ti, tj = char_span_to_token_range(tok, start, end)
        return AlignmentResult(
            status=status,
            start=start,
            end=end,
            token_start=ti,
            token_end=tj,
            similarity=sim,
            method=method,
            n_candidates=n_cand,
            recovered_text=source[start:end],
        )


# ---------------------------------------------------------------- helpers
def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _find_all(haystack: str, needle: str) -> list[int]:
    out = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def _choose_occurrence(occ: list[int], length: int, hint_pos: Optional[int], prev_end: int) -> int:
    if hint_pos is not None:
        return min(occ, key=lambda s: (abs(s - hint_pos), s))
    after = [s for s in occ if s >= prev_end]
    return after[0] if after else occ[0]


def _map_source_pos_to_norm(src_map: list[int], pos: Optional[int]) -> Optional[int]:
    if pos is None:
        return None
    # first normalised index whose source index >= pos
    lo, hi = 0, len(src_map)
    while lo < hi:
        mid = (lo + hi) // 2
        if src_map[mid] < pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def exact_only_aligner(tokenizer: Tokenizer | None = None) -> TextAligner:
    """The A5 ablation: verbatim matching only (no normalisation, no fuzzy recovery, no snapping)."""
    return TextAligner(tokenizer=tokenizer, allow_normalized=False, allow_fuzzy=False, boundary_snap=False)
