"""Stage 5 - Quality Validation (Recorder) and the bounded refinement loop.

Validation criteria (Section 3.6): XML validity, label legality, span non-emptiness and
recoverable source-text alignment.  Blocking issues trigger at most ``R`` refinement
re-queries (Algorithm 1, line 6).  Rejected spans are *flagged* and kept in the record so
that the evaluation can count them (``rejected_span_policy``) - they are never silently
promoted to accepted annotations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .alignment import AlignmentResult, TextAligner
from .analyzer import ParsedOutput, RawAnnotation
from .preprocessor import ProcessedText


@dataclass
class ValidationIssue:
    code: str
    message: str
    index: Optional[int] = None
    blocking: bool = True

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "index": self.index, "blocking": self.blocking}


@dataclass
class AlignedSpan:
    order: int
    text: str
    label: Optional[str]
    label_raw: str
    status: str  # exact | normalized | fuzzy | rejected | illegal_label | empty
    start: Optional[int] = None  # processed-text offsets
    end: Optional[int] = None
    original_start: Optional[int] = None  # original-text offsets (evaluation coordinates)
    original_end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    similarity: float = 0.0
    method: str = "none"
    index_hint: Optional[tuple[int, int]] = None
    recovered_text: str = ""

    @property
    def recovered(self) -> bool:
        return self.status in {"exact", "normalized", "fuzzy"}

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "text": self.text,
            "label": self.label,
            "label_raw": self.label_raw,
            "status": self.status,
            "start": self.start,
            "end": self.end,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "similarity": round(self.similarity, 4),
            "method": self.method,
            "index_hint": list(self.index_hint) if self.index_hint else None,
            "recovered_text": self.recovered_text,
        }


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    n_annotations: int = 0
    n_illegal_label: int = 0
    n_empty: int = 0
    n_unrecoverable: int = 0
    n_duplicates: int = 0

    @property
    def blocking(self) -> bool:
        return any(i.blocking for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "blocking": self.blocking,
            "n_annotations": self.n_annotations,
            "n_illegal_label": self.n_illegal_label,
            "n_empty": self.n_empty,
            "n_unrecoverable": self.n_unrecoverable,
            "n_duplicates": self.n_duplicates,
            "issues": [i.to_dict() for i in self.issues],
        }

    def feedback(self, max_items: int = 8) -> str:
        lines = []
        for i in self.issues[:max_items]:
            if i.blocking:
                lines.append(f"- {i.message}")
        if len(self.issues) > max_items:
            lines.append(f"- ... and {len(self.issues) - max_items} further issues")
        return "\n".join(lines) if lines else "- unspecified validation failure"


class Validator:
    def __init__(self, allowed_labels: list[str], aligner: TextAligner, validate_structure: bool = True, validate_labels: bool = True, validate_alignment: bool = True):
        self.allowed = set(allowed_labels)
        self.aligner = aligner
        self.validate_structure = validate_structure
        self.validate_labels = validate_labels
        self.validate_alignment = validate_alignment

    def validate(self, parsed: ParsedOutput, processed: ProcessedText) -> tuple[ValidationReport, list[AlignedSpan]]:
        report = ValidationReport(ok=True, n_annotations=len(parsed.annotations))
        if self.validate_structure:
            for issue in parsed.structural_issues:
                code = issue.split(":", 1)[0]
                report.issues.append(ValidationIssue(code, issue, blocking=code not in {"INLINE_TAG_FORMAT"}))
        spans: list[AlignedSpan] = []
        seen: set[tuple] = set()
        prev_end = 0
        for ann in parsed.annotations:
            sp = self._align_one(ann, processed, prev_end)
            if sp.status == "empty":
                report.n_empty += 1
                report.issues.append(ValidationIssue("EMPTY_SPAN", f"annotation {ann.order} has an empty <text>", ann.order))
            if self.validate_labels and ann.label is None:
                report.n_illegal_label += 1
                sp.status = "illegal_label" if sp.status in {"exact", "normalized", "fuzzy"} else sp.status
                report.issues.append(ValidationIssue("ILLEGAL_LABEL", f"annotation {ann.order} uses the label '{ann.label_raw}', which is not one of: {', '.join(sorted(self.allowed))}", ann.order))
            elif ann.label is not None and ann.label not in self.allowed and self.validate_labels:
                report.n_illegal_label += 1
                sp.status = "illegal_label"
                report.issues.append(ValidationIssue("ILLEGAL_LABEL", f"annotation {ann.order} uses the label '{ann.label_raw}', which is not permitted", ann.order))
            if sp.status == "rejected":
                report.n_unrecoverable += 1
                if self.validate_alignment:
                    report.issues.append(ValidationIssue("UNRECOVERABLE_SPAN", f"annotation {ann.order}: the text \"{_short(ann.text)}\" cannot be located in the reviewer comment; copy the span verbatim", ann.order))
            key = (sp.original_start, sp.original_end, sp.label)
            if sp.recovered and key in seen:
                report.n_duplicates += 1
                report.issues.append(ValidationIssue("DUPLICATE_SPAN", f"annotation {ann.order} duplicates an earlier span", ann.order, blocking=False))
                continue
            if sp.recovered:
                seen.add(key)
                prev_end = sp.end or prev_end
            spans.append(sp)
        report.ok = not report.blocking
        return report, spans

    def _align_one(self, ann: RawAnnotation, processed: ProcessedText, prev_end: int) -> AlignedSpan:
        if not ann.text.strip():
            return AlignedSpan(ann.order, ann.text, ann.label, ann.label_raw, "empty", index_hint=ann.index_hint)
        hint = ann.index_hint[0] if ann.index_hint else None
        res: AlignmentResult = self.aligner.align(processed.text, ann.text, hint_pos=hint, prev_end=prev_end)
        if not res.recovered:
            return AlignedSpan(ann.order, ann.text, ann.label, ann.label_raw, "rejected", similarity=res.similarity, method=res.method, index_hint=ann.index_hint)
        os_, oe = processed.map_span(res.start, res.end)
        return AlignedSpan(
            order=ann.order,
            text=ann.text,
            label=ann.label,
            label_raw=ann.label_raw,
            status=res.status,
            start=res.start,
            end=res.end,
            original_start=os_,
            original_end=oe,
            token_start=res.token_start,
            token_end=res.token_end,
            similarity=res.similarity,
            method=res.method,
            index_hint=ann.index_hint,
            recovered_text=res.recovered_text,
        )


def _short(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


class Recorder:
    """Assemble the final annotation record according to the validation policy."""

    def __init__(self, keep_illegal_labels: bool = False):
        self.keep_illegal_labels = keep_illegal_labels

    def finalize(self, spans: list[AlignedSpan]) -> list[AlignedSpan]:
        out = []
        for sp in spans:
            if sp.status == "empty":
                continue
            if sp.status == "illegal_label" and not self.keep_illegal_labels:
                continue
            out.append(sp)
        return out
