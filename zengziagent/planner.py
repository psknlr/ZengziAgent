"""Stage 1 - Task Analysis (Planner).

The Planner converts an externally defined annotation schema, a demonstration pool and
the *unlabelled* target corpus into the structured task specification
``S_D = (U_D, L_D, G_D, E_D, C_D, O_D, V_D)`` (Table 1).  It never creates labels and never
sees the gold labels of the evaluation records: demonstrations are drawn from a separate
pool (SubstanReview train split / held-out annotated eLife reports).

Everything is deterministic given the seed, so the same inputs yield byte-identical
specifications (the specification hash is stored in every run manifest).
"""
from __future__ import annotations

import re
import statistics
from typing import Optional

from .preprocessor import segment_sentences
from .schema import AnnotationSchema, Demonstration, LabelDefinition, ReviewRecord, TaskSpecification
from .tokenization import get_tokenizer
from .utils import utc_now

_NUMBERED_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")
_REVIEWER_RE = re.compile(r"(?i)reviewer\s*#?\s*\d")
_ROUND_RE = re.compile(r"(?i)\b(round\s*\d|revised (?:version|manuscript)|resubmission|essential revisions)\b")

OUTPUT_SCHEMA = {
    "format": "xml_annotation_blocks",
    "root": "annotations",
    "block": "annotation",
    "fields": {"text": "span copied verbatim from the reviewer comment", "label": "one permitted label", "index": "optional start-end character offsets (position hint only)"},
    "example": "<annotations>\n  <annotation>\n    <text>The paper is well-written and well-structured.</text>\n    <index>244-291</index>\n    <label>Eval_pos</label>\n  </annotation>\n</annotations>",
}


class Planner:
    def __init__(self, schema: AnnotationSchema, n_demonstrations: int = 3, seed: int = 0, tokenizer=None):
        self.schema = schema
        self.n_demonstrations = n_demonstrations
        self.seed = seed
        self.tokenizer = tokenizer or get_tokenizer("auto")

    # ------------------------------------------------------------------ Stage 1
    def build(
        self,
        dataset: str,
        target_records: list[ReviewRecord],
        demo_pool: Optional[list[ReviewRecord]] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> TaskSpecification:
        profile_cfg = self.schema.dataset_profiles.get(dataset, {})
        # 1. Requirement analysis
        unit = {
            "definition": self.schema.unit_definition,
            "granularity": "span (sentence or clause; may cross sentence boundaries)",
            "one_label_per_span": True,
        }
        labels = list(self.schema.labels)
        validation_rules = {
            "allowed_labels": [l.name for l in labels],
            "xml_root": OUTPUT_SCHEMA["root"],
            "required_fields": ["text", "label"],
            "non_empty_span": True,
            "verbatim_copy_required": True,
            "recoverable_source_alignment": True,
            "occurrence_suffixes_are_metadata": True,
        }
        # 2. Dataset characterisation (from unlabelled target text + configured profile)
        computed = self._characterize(target_records)
        constraints = {
            "dataset": dataset,
            "display_name": profile_cfg.get("display_name", dataset),
            "domain": profile_cfg.get("domain", "scientific peer review"),
            "structure": profile_cfg.get("structure", ""),
            "notes": list(profile_cfg.get("notes", [])),
            "computed_profile": computed,
            "annotation_target": "reviewer text only",
        }
        if computed.get("reviewer_header_fraction", 0) > 0.2 or computed.get("round_marker_fraction", 0) > 0.2:
            constraints["notes"].append("Reports are organised in reviewer sections and/or review rounds; annotate each reviewer's text independently and do not annotate section headers.")
        if computed.get("mean_tokens", 0) > 600:
            constraints["notes"].append("Reports are long; scan the whole text and annotate every evaluative statement, not only the first paragraph.")
        # 3. Demonstration selection (coverage-greedy, deterministic)
        exclude = set(exclude_ids or set()) | {r.review_id for r in target_records}
        demos = self._select_demonstrations(demo_pool or [], exclude)
        guidelines = list(self.schema.guideline_notes) + [f"Dataset note: {n}" for n in constraints["notes"]]
        spec = TaskSpecification(
            dataset=dataset,
            objective=self.schema.objective,
            unit=unit,
            labels=labels,
            guidelines=guidelines,
            demonstrations=demos,
            constraints=constraints,
            output_schema=OUTPUT_SCHEMA,
            validation_rules=validation_rules,
            kind="planner",
            provenance={
                "schema": self.schema.name,
                "n_demonstrations_requested": self.n_demonstrations,
                "demo_pool_size": len(demo_pool or []),
                "seed": self.seed,
                "created_at": utc_now(),
            },
        )
        return spec

    @staticmethod
    def generic(schema: AnnotationSchema, dataset: str) -> TaskSpecification:
        """The *w/o Planner* replacement (ablation A1): labels, objective and output format
        only - no dataset profile, no demonstrations, no dataset-specific constraints."""
        return TaskSpecification(
            dataset=dataset,
            objective=schema.objective,
            unit={"definition": "a text span of the reviewer comment"},
            labels=[LabelDefinition(name=l.name, definition=l.definition) for l in schema.labels],
            guidelines=["Copy every span verbatim from the reviewer comment.", "Use only the permitted labels."],
            demonstrations=[],
            constraints={"dataset": dataset},
            output_schema=OUTPUT_SCHEMA,
            validation_rules={"allowed_labels": [l.name for l in schema.labels], "xml_root": OUTPUT_SCHEMA["root"], "required_fields": ["text", "label"], "non_empty_span": True, "verbatim_copy_required": True, "recoverable_source_alignment": True},
            kind="generic",
            provenance={"schema": schema.name, "created_at": utc_now()},
        )

    # ----------------------------------------------------------------- helpers
    def _characterize(self, records: list[ReviewRecord]) -> dict:
        if not records:
            return {}
        toks = [len(self.tokenizer.spans(r.text)) for r in records]
        sents = [len(segment_sentences(r.text)) for r in records]
        paras = [max(1, r.text.count("\n") + 1) for r in records]
        return {
            "n_records": len(records),
            "mean_tokens": round(statistics.mean(toks), 1),
            "median_tokens": statistics.median(toks),
            "mean_sentences": round(statistics.mean(sents), 1),
            "mean_paragraphs": round(statistics.mean(paras), 1),
            "numbered_item_fraction": round(sum(1 for r in records if _NUMBERED_RE.search(r.text)) / len(records), 3),
            "reviewer_header_fraction": round(sum(1 for r in records if _REVIEWER_RE.search(r.text)) / len(records), 3),
            "round_marker_fraction": round(sum(1 for r in records if _ROUND_RE.search(r.text)) / len(records), 3),
        }

    def _select_demonstrations(self, pool: list[ReviewRecord], exclude: set[str]) -> list[Demonstration]:
        cands = [r for r in pool if r.review_id not in exclude and len(r.gold_spans) >= 2]
        if not cands or self.n_demonstrations <= 0:
            return []
        needed = {l.name for l in self.schema.labels}
        # deterministic ordering: shorter reviews first, ties by id
        cands.sort(key=lambda r: (len(r.text), r.review_id))
        chosen: list[ReviewRecord] = []
        covered: set[str] = set()
        while len(chosen) < self.n_demonstrations and cands:
            if covered >= needed:
                best = cands[0]  # coverage complete: add the shortest remaining review
            else:
                best = max(cands, key=lambda r: (len({g.label for g in r.gold_spans} - covered), -len(r.text)))
            chosen.append(best)
            covered |= {g.label for g in best.gold_spans}
            cands.remove(best)
        return [Demonstration(review_id=r.review_id, text=r.text, spans=list(r.gold_spans)) for r in chosen]
