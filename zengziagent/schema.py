"""Core data structures.

The formal objects of the manuscript are mirrored here:

* ``TaskSpecification`` is the Planner output
  ``S_D = (U_D, L_D, G_D, E_D, C_D, O_D, V_D)``.
* ``PipelineConfig`` encodes which stages are active; the named ablation
  configurations (F, B0, A1-A6) live in :mod:`zengziagent.experiments.configs`.
* ``ReviewRecord`` / ``GoldSpan`` / ``AnnotationUnit`` describe evaluation data.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .utils import sha256_json

#: The mapped five-label evaluation scheme (Table 5).
LABELS: tuple[str, ...] = ("Eval_pos", "Eval_neg", "Jus_pos", "Jus_neg", "Major_Claim")

#: Canonicalisation of label surface forms produced by datasets or LLMs.
_LABEL_ALIASES = {
    "eval_pos": "Eval_pos",
    "eval_neg": "Eval_neg",
    "jus_pos": "Jus_pos",
    "jus_neg": "Jus_neg",
    "major_claim": "Major_Claim",
    "majorclaim": "Major_Claim",
    "major claim": "Major_Claim",
    "positive_claim": "Eval_pos",
    "negative_claim": "Eval_neg",
    "positive_evidence": "Jus_pos",
    "negative_evidence": "Jus_neg",
    "claim_pos": "Eval_pos",
    "claim_neg": "Eval_neg",
    "evidence_pos": "Jus_pos",
    "evidence_neg": "Jus_neg",
}
_SUFFIX_RE = re.compile(r"[_\-\s]?(\d+)$")


def canonical_label(raw: str) -> tuple[Optional[str], Optional[int]]:
    """Map a raw label string to ``(canonical_label, occurrence_number)``.

    ``"Jus_neg_2"`` -> ``("Jus_neg", 2)``; ``"Major_claim"`` -> ``("Major_Claim", None)``.
    Unknown labels return ``(None, None)`` so that validation can flag them.
    Occurrence suffixes are metadata (Section 4.1.3), never separate labels.
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    occ = None
    m = _SUFFIX_RE.search(s)
    if m:
        occ = int(m.group(1))
        s = s[: m.start()]
    key = s.strip().lower().replace("-", "_").replace(" ", "_")
    canon = _LABEL_ALIASES.get(key)
    if canon is None and s in LABELS:
        canon = s
    return canon, occ


# ---------------------------------------------------------------------------
# Evaluation data
# ---------------------------------------------------------------------------
@dataclass
class GoldSpan:
    start: int
    end: int
    label: str
    occurrence: Optional[int] = None
    raw_label: Optional[str] = None

    @property
    def text_slice(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewRecord:
    """One reviewer comment (a SubstanReview review or an eLife reviewer report)."""

    review_id: str
    dataset: str
    text: str
    gold_spans: list[GoldSpan] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def gold_text(self, span: GoldSpan) -> str:
        return self.text[span.start : span.end]

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "dataset": self.dataset,
            "text": self.text,
            "gold_spans": [s.to_dict() for s in self.gold_spans],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewRecord":
        return cls(
            review_id=str(d["review_id"]),
            dataset=d["dataset"],
            text=d["text"],
            gold_spans=[GoldSpan(**s) for s in d.get("gold_spans", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass
class AnnotationUnit:
    """An evaluated annotation unit (Section 4.3.3).

    In the default ``gold_span`` unit mode each gold span is one unit carrying one
    primary gold label; unit-level accuracy asks whether the system assigns that
    label to the unit, while span-level P/R/F1 are computed from all spans.
    """

    unit_id: str
    review_id: str
    dataset: str
    start: int
    end: int
    gold_label: str
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Annotation schema (externally defined; loaded from configs/schema/*.yaml)
# ---------------------------------------------------------------------------
@dataclass
class LabelDefinition:
    name: str
    definition: str
    polarity: str = ""
    example: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnotationSchema:
    """The externally supplied annotation protocol: labels, guideline, output tags."""

    name: str
    labels: list[LabelDefinition]
    guideline_notes: list[str]
    objective: str
    unit_definition: str
    output_format: str = "xml_annotation_blocks"
    dataset_profiles: dict[str, dict] = field(default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "labels": [l.to_dict() for l in self.labels],
            "guideline_notes": list(self.guideline_notes),
            "objective": self.objective,
            "unit_definition": self.unit_definition,
            "output_format": self.output_format,
            "dataset_profiles": self.dataset_profiles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationSchema":
        return cls(
            name=d["name"],
            labels=[LabelDefinition(**l) for l in d["labels"]],
            guideline_notes=list(d.get("guideline_notes", [])),
            objective=d["objective"],
            unit_definition=d["unit_definition"],
            output_format=d.get("output_format", "xml_annotation_blocks"),
            dataset_profiles=d.get("dataset_profiles", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AnnotationSchema":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Planner output: the structured task specification S_D
# ---------------------------------------------------------------------------
@dataclass
class Demonstration:
    review_id: str
    text: str
    spans: list[GoldSpan]

    def to_dict(self) -> dict:
        return {"review_id": self.review_id, "text": self.text, "spans": [s.to_dict() for s in self.spans]}


@dataclass
class TaskSpecification:
    """``S_D = (U_D, L_D, G_D, E_D, C_D, O_D, V_D)`` (Section 3.2 of the revised manuscript).

    Task adaptation is the *deterministic compilation* of this externally defined
    specification into dataset- and backend-conditioned prompts; nothing here is
    learned online or updated by the LLM.
    """

    dataset: str
    unit: dict  # U_D: annotation unit definition
    labels: list[LabelDefinition]  # L_D
    guidelines: list[str]  # G_D
    demonstrations: list[Demonstration]  # E_D
    constraints: dict  # C_D: dataset profile + dataset/context constraints
    output_schema: dict  # O_D
    validation_rules: dict  # V_D
    objective: str = ""
    kind: str = "planner"  # "planner" (full Stage 1) or "generic" (A1: w/o Planner)
    provenance: dict = field(default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "kind": self.kind,
            "objective": self.objective,
            "unit": self.unit,
            "labels": [l.to_dict() for l in self.labels],
            "guidelines": list(self.guidelines),
            "demonstrations": [d.to_dict() for d in self.demonstrations],
            "constraints": self.constraints,
            "output_schema": self.output_schema,
            "validation_rules": self.validation_rules,
            "provenance": self.provenance,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def hash(self) -> str:
        d = self.to_dict()
        d.pop("provenance", None)
        return sha256_json(d)[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpecification":
        return cls(
            dataset=d["dataset"],
            unit=d["unit"],
            labels=[LabelDefinition(**l) for l in d["labels"]],
            guidelines=list(d.get("guidelines", [])),
            demonstrations=[
                Demonstration(review_id=x["review_id"], text=x["text"], spans=[GoldSpan(**s) for s in x["spans"]])
                for x in d.get("demonstrations", [])
            ],
            constraints=d.get("constraints", {}),
            output_schema=d.get("output_schema", {}),
            validation_rules=d.get("validation_rules", {}),
            objective=d.get("objective", ""),
            kind=d.get("kind", "planner"),
            provenance=d.get("provenance", {}),
        )


# ---------------------------------------------------------------------------
# Pipeline configuration (ablation switches)
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Which stages are active.  Each flag has an explicit *replacement* when off
    (documented in ``docs/ABLATION_DESIGN.md`` and ``experiments/configs.py``)."""

    config_id: str = "F"
    name: str = "Full ZengziAgent"
    use_planner: bool = True  # A1 off -> generic specification (labels + format only)
    use_adaptive_prompting: bool = True  # A2 off -> Fixed Instruction Baseline prompt
    use_preprocessing: bool = True  # A3 off -> raw text, identity offset map
    use_validation: bool = True  # A4 off -> no schema validation, no refinement
    use_refinement: bool = True  # A6 off -> validation-only, no re-query
    use_alignment: bool = True  # A5 off -> exact string match only
    max_refinement_retries: int = 2  # R in Algorithm 1
    alignment_similarity_threshold: float = 0.80  # character-level acceptance threshold
    prompt_synthesis: str = "deterministic"  # "deterministic" | "llm" (Table 2 meta-prompt)
    n_demonstrations: int = 3
    reflection_rounds: int = 1  # 1 = single pass (R1); 2 = add reflection pass (R2)
    description: str = ""
    replacement: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return sha256_json(self.to_dict())[:12]


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)
