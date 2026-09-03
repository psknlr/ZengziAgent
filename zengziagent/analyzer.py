"""Stage 4 - Annotation Generation (Analyzer): LLM call + tolerant XML parsing."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Optional
from xml.etree import ElementTree as ET

from .actor import PromptBundle
from .llm.base import ChatMessage, LLMBackend, LLMResponse
from .schema import LABELS, SamplingParams, canonical_label

_FENCE_RE = re.compile(r"```(?:xml)?\s*(.*?)```", re.S)
_ROOT_OPEN_RE = re.compile(r"<annotations\b[^>]*>", re.I)
_ROOT_CLOSE_RE = re.compile(r"</annotations\s*>", re.I)
_BLOCK_RE = re.compile(r"<annotation\b[^>]*>(.*?)</annotation\s*>", re.S | re.I)
_OPEN_BLOCK_RE = re.compile(r"<annotation\b[^>]*>", re.I)
_CLOSE_BLOCK_RE = re.compile(r"</annotation\s*>", re.I)
_TEXT_RE = re.compile(r"<text\b[^>]*>(.*?)</text\s*>", re.S | re.I)
_LABEL_RE = re.compile(r"<label\b[^>]*>(.*?)</label\s*>", re.S | re.I)
_INDEX_RE = re.compile(r"<index\b[^>]*>\s*(\d+)\s*[-–,:]\s*(\d+)\s*</index\s*>", re.S | re.I)
_INLINE_RE = re.compile(r"<(Eval_pos|Eval_neg|Jus_pos|Jus_neg|Major_Claim|Major_claim)(?:_\d+)?\b[^>]*>(.*?)</\1(?:_\d+)?\s*>", re.S)


@dataclass
class RawAnnotation:
    order: int
    text: str
    label_raw: str
    label: Optional[str]
    occurrence: Optional[int] = None
    index_hint: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict:
        return {"order": self.order, "text": self.text, "label_raw": self.label_raw, "label": self.label, "occurrence": self.occurrence, "index_hint": list(self.index_hint) if self.index_hint else None}


@dataclass
class ParsedOutput:
    annotations: list[RawAnnotation] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    strict_well_formed: bool = False
    n_open_blocks: int = 0
    n_close_blocks: int = 0
    format: str = "annotation_blocks"

    @property
    def well_formed(self) -> bool:
        return not self.structural_issues


def _unescape(s: str) -> str:
    """Decode named *and* numeric character references (``&#39;``, ``&#x2019;`` ...)."""
    return html.unescape(s).strip()


def parse_xml_annotations(raw: str) -> ParsedOutput:
    """Tolerant parser for the annotation-block XML format (see ``planner.OUTPUT_SCHEMA``)."""
    out = ParsedOutput()
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    has_open = bool(_ROOT_OPEN_RE.search(text))
    has_close = bool(_ROOT_CLOSE_RE.search(text))
    out.n_open_blocks = len(_OPEN_BLOCK_RE.findall(text))
    out.n_close_blocks = len(_CLOSE_BLOCK_RE.findall(text))
    if not text:
        out.structural_issues.append("EMPTY_OUTPUT: the response contained no XML")
        return out
    if not has_open:
        out.structural_issues.append("MISSING_ROOT: no <annotations> root element")
    elif not has_close:
        out.structural_issues.append("MISSING_ROOT_CLOSE: </annotations> closing tag missing")
    if out.n_open_blocks != out.n_close_blocks:
        out.structural_issues.append(f"UNBALANCED_BLOCKS: {out.n_open_blocks} <annotation> vs {out.n_close_blocks} </annotation>")
    blocks = _BLOCK_RE.findall(text)
    if not blocks and out.n_open_blocks == 0:
        inline = _INLINE_RE.findall(text)
        if inline:
            out.format = "inline_tags"
            out.structural_issues.append("INLINE_TAG_FORMAT: inline <Label>span</Label> tags used instead of annotation blocks")
            for k, (lab, span) in enumerate(inline):
                canon, occ = canonical_label(lab)
                out.annotations.append(RawAnnotation(k, _unescape(span), lab, canon, occ))
            return out
    for k, block in enumerate(blocks):
        tm = _TEXT_RE.search(block)
        lm = _LABEL_RE.search(block)
        im = _INDEX_RE.search(block)
        if tm is None or lm is None:
            out.structural_issues.append(f"INCOMPLETE_BLOCK: annotation {k} lacks <text> or <label>")
            continue
        label_raw = _unescape(lm.group(1))
        canon, occ = canonical_label(label_raw)
        hint = (int(im.group(1)), int(im.group(2))) if im else None
        out.annotations.append(RawAnnotation(k, _unescape(tm.group(1)), label_raw, canon, occ, hint))
    # strict well-formedness (informational): escape bare ampersands and try a real XML parser
    try:
        candidate = text[_ROOT_OPEN_RE.search(text).start() :] if has_open else text
        candidate = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", candidate)
        ET.fromstring(candidate)
        out.strict_well_formed = True
    except Exception:
        out.strict_well_formed = False
    return out


class Analyzer:
    def __init__(self, backend: LLMBackend, params: SamplingParams | None = None, reflection_prompt: Optional[str] = None):
        self.backend = backend
        self.params = params or SamplingParams()
        self.reflection_prompt = reflection_prompt

    def annotate(
        self,
        bundle: PromptBundle,
        review_id: str,
        text: str,
        dataset: str = "",
        run_index: int = 0,
        validation_feedback: Optional[str] = None,
        previous_xml: Optional[str] = None,
        previous_output: Optional[str] = None,
    ) -> tuple[LLMResponse, ParsedOutput]:
        user = bundle.user_message(review_id, text, dataset)
        messages = [ChatMessage("system", bundle.system_prompt)]
        if previous_xml is not None:  # reflection round (R2)
            user = (self.reflection_prompt or "Re-examine the previous annotation and return the complete revised XML.") + "\n\n" + user + f"\n<previous_annotation>\n{previous_xml}\n</previous_annotation>"
            messages.append(ChatMessage("user", user))
        elif validation_feedback is not None:  # refinement round (Stage 5 feedback loop)
            messages.append(ChatMessage("user", user))
            note = ""
            if previous_output is not None and previous_output.strip():
                messages.append(ChatMessage("assistant", previous_output))
            else:  # empty assistant turns are rejected by some providers
                note = "(Your previous response was empty.) "
            messages.append(ChatMessage("user", note + "Your previous output failed validation:\n" + validation_feedback + "\nReturn the complete corrected XML only. Every <text> must be copied verbatim from the reviewer comment and every <label> must be one of the permitted labels."))
        else:
            messages.append(ChatMessage("user", user))
        resp = self.backend.complete(messages, self.params, run_index=run_index, tag=f"{dataset}:{review_id}")
        parsed = parse_xml_annotations(resp.text)
        return resp, parsed
