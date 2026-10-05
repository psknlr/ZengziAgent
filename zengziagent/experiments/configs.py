"""Named pipeline configurations for the component-contribution study (Section 6).

Each ablation states *what replaces the removed component* so that the comparison is
well defined (reviewer point 1.2).
"""
from __future__ import annotations

from dataclasses import replace

from ..schema import PipelineConfig

FULL = PipelineConfig(
    config_id="F",
    name="Full ZengziAgent",
    description="Planner -> adaptive prompt compilation -> deterministic preprocessing -> Analyzer -> validation with bounded refinement -> Text Alignment Tool.",
    replacement="-",
)

ABLATIONS: dict[str, PipelineConfig] = {
    "F": FULL,
    "B0": replace(
        FULL,
        config_id="B0",
        name="Direct LLM + Fixed Instruction Baseline",
        use_planner=False,
        use_adaptive_prompting=False,
        use_preprocessing=False,
        use_validation=False,
        use_refinement=False,
        use_alignment=False,
        description="A single call with the static Fixed Instruction Baseline prompt (objective, permitted labels, output format only) on the raw text; output parsed once, spans accepted only if they occur verbatim in the source.",
        replacement="FIB prompt; no specification, no preprocessing, no validation/refinement, exact matching only.",
    ),
    "A1": replace(
        FULL,
        config_id="A1",
        name="w/o Planner",
        use_planner=False,
        description="Structured task specification replaced by a generic specification (label names and one-line definitions, objective, output format). No dataset profile, no demonstration selection, no dataset-specific constraints or validation notes; the Actor still compiles and renders the prompt for the backend.",
        replacement="Generic specification S_generic (labels + format only).",
    ),
    "A2": replace(
        FULL,
        config_id="A2",
        name="w/o Task-Adaptive Prompting",
        use_adaptive_prompting=False,
        description="The compiled, backend-rendered prompt P_{D,M} is replaced by the Fixed Instruction Baseline P_fixed (Supplementary): a dataset-independent instruction containing only the annotation objective, the permitted labels and the output format; no dataset profile, no examples, no context-specific instructions, no backend-specific rendering. The Planner specification is archived but not consumed; all other stages unchanged.",
        replacement="P_fixed (Fixed Instruction Baseline) instead of P_adaptive(S_D, M).",
    ),
    "A3": replace(
        FULL,
        config_id="A3",
        name="w/o deterministic preprocessing",
        use_preprocessing=False,
        description="Raw reviewer text (no Unicode/whitespace normalisation, no markup removal) is passed to the Analyzer and used as the alignment reference.",
        replacement="Identity preprocessing.",
    ),
    "A4": replace(
        FULL,
        config_id="A4",
        name="w/o Recorder validation/refinement",
        use_validation=False,
        use_refinement=False,
        description="No schema validation and no repair loop: the first response is parsed tolerantly; illegal labels and unrecoverable spans are kept as predictions (they count as false positives).",
        replacement="Single-pass tolerant parsing.",
    ),
    "A5": replace(
        FULL,
        config_id="A5",
        name="w/o Text Alignment",
        use_alignment=False,
        description="Only exact (verbatim) source-span matching is allowed; normalised matching, fuzzy recovery, boundary correction and token-range remapping are disabled. Non-verbatim spans are rejected and counted as false positives.",
        replacement="Exact string matching only.",
    ),
    "A6": replace(
        FULL,
        config_id="A6",
        name="validation only (w/o refinement retry)",
        use_refinement=False,
        description="The Recorder validates and flags/drops invalid items but never re-queries the LLM, separating the benefit of checking from the benefit of feedback repair.",
        replacement="Validation without the feedback loop (R = 0).",
    ),
}

ORDER = ["F", "B0", "A1", "A2", "A3", "A4", "A5", "A6"]


def get_config(config_id: str, **overrides) -> PipelineConfig:
    if config_id not in ABLATIONS:
        raise KeyError(f"unknown configuration '{config_id}'; known: {ORDER}")
    cfg = ABLATIONS[config_id]
    return replace(cfg, **overrides) if overrides else cfg


def ablation_table() -> list[dict]:
    rows = []
    for cid in ORDER:
        c = ABLATIONS[cid]
        rows.append(
            {
                "ID": cid,
                "Configuration": c.name,
                "Planner": c.use_planner,
                "Adaptive prompting": c.use_adaptive_prompting,
                "Preprocessing": c.use_preprocessing,
                "Validation": c.use_validation,
                "Refinement": c.use_refinement,
                "Alignment": c.use_alignment,
                "Replacement": c.replacement,
                "Description": c.description,
            }
        )
    return rows
