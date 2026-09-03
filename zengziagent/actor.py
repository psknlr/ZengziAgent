"""Stage 2 - Adaptive Prompting (Actor).

``P_{D,M} = R_M(C(S_D))``: the *task-specification compiler* ``C`` turns the Planner's
specification into a backend-neutral prompt intermediate representation (ordered
sections), and the *backend renderer* ``R_M`` serialises it in the format each LLM family
follows best (XML-tagged sections for Claude, Markdown headings for GPT models, numbered
plain sections for Gemini, Markdown for MiniMax / others).  Optionally the guideline text
can be synthesised by the LLM itself with the Table 2 meta-prompt (``prompt_synthesis="llm"``);
the synthesised text is cached and logged so that it can be released verbatim.

The *Fixed Instruction Baseline* (FIB) used by ablations A2 and B0 is the static,
dataset-independent prompt in ``prompts/fixed_instruction_baseline.txt``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .llm.base import ChatMessage, LLMBackend
from .schema import Demonstration, PipelineConfig, SamplingParams, TaskSpecification
from .utils import repo_root, sha256_text

USER_TEMPLATE = (
    "Annotate the following reviewer comment{dataset_note}. Return only the XML.\n"
    '<review id="{review_id}">\n{text}\n</review>'
)
FIB_USER_TEMPLATE = '<review id="{review_id}">\n{text}\n</review>'


@dataclass
class PromptSection:
    section_id: str
    title: str
    body: str
    kind: str = "text"  # text | list | xml


@dataclass
class PromptIR:
    sections: list[PromptSection]
    spec_hash: str
    spec_kind: str


@dataclass
class PromptBundle:
    system_prompt: str
    user_template: str
    mode: str  # "adaptive" | "fixed"
    family: str
    spec_hash: Optional[str]
    prompt_hash: str
    provenance: dict = field(default_factory=dict)

    def user_message(self, review_id: str, text: str, dataset: str = "") -> str:
        note = f" from the {dataset} corpus" if (dataset and self.mode == "adaptive") else ""
        return self.user_template.format(review_id=review_id, text=text, dataset_note=note)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "family": self.family,
            "spec_hash": self.spec_hash,
            "prompt_hash": self.prompt_hash,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "provenance": self.provenance,
        }


def render_demonstrations(demos: list[Demonstration], max_chars: int = 2500) -> str:
    """Render demonstrations as XML examples (Table 9 format)."""
    parts = []
    for d in demos:
        text = d.text if len(d.text) <= max_chars else d.text[:max_chars]
        blocks = []
        for s in d.spans:
            if s.end <= len(text):
                blocks.append(f"  <annotation>\n    <text>{d.text[s.start:s.end]}</text>\n    <index>{s.start}-{s.end}</index>\n    <label>{s.label}</label>\n  </annotation>")
        parts.append(f"<example>\n<review>\n{text}\n</review>\n<annotations>\n" + "\n".join(blocks) + "\n</annotations>\n</example>")
    return "\n\n".join(parts)


class PromptCompiler:
    """``C``: TaskSpecification -> PromptIR (backend neutral)."""

    def compile(self, spec: TaskSpecification, synthesized_guidelines: Optional[str] = None) -> PromptIR:
        sections: list[PromptSection] = []
        sections.append(PromptSection("role", "Role", "You are an expert annotator of scientific peer-review comments. You annotate what reviewers say and how they evaluate manuscripts; you do not judge review quality or the manuscript itself."))
        sections.append(PromptSection("objective", "Annotation objective", spec.objective))
        sections.append(PromptSection("unit", "Annotation unit", spec.unit.get("definition", "")))
        label_lines = []
        for l in spec.labels:
            line = f"{l.name}: {l.definition}"
            if l.example:
                line += f' Example: "{l.example}"'
            label_lines.append(line)
        sections.append(PromptSection("labels", "Permitted labels (use exactly these strings)", "\n".join(label_lines), kind="list"))
        if spec.kind == "planner":
            c = spec.constraints
            ctx = [f"Corpus: {c.get('display_name', spec.dataset)}", f"Domain: {c.get('domain', '')}", f"Structure: {c.get('structure', '')}"]
            prof = c.get("computed_profile") or {}
            if prof:
                ctx.append(f"Typical length: about {prof.get('mean_tokens', '?')} tokens and {prof.get('mean_sentences', '?')} sentences per reviewer comment.")
            for n in c.get("notes", []):
                ctx.append(f"Note: {n}")
            sections.append(PromptSection("context", "Corpus context and constraints", "\n".join(ctx), kind="list"))
        if synthesized_guidelines:
            sections.append(PromptSection("synthesized", "Task-specific instructions (synthesised from the task specification)", synthesized_guidelines))
        if spec.guidelines:
            sections.append(PromptSection("guidelines", "Annotation guidelines", "\n".join(f"- {g}" for g in spec.guidelines), kind="list"))
        if spec.demonstrations:
            sections.append(PromptSection("demonstrations", "Annotated examples", render_demonstrations(spec.demonstrations), kind="xml"))
        out = spec.output_schema
        sections.append(PromptSection("output", "Output format", "Return only XML in exactly this form (one <annotation> block per span; <index> is optional and used only as a position hint):\n" + out.get("example", ""), kind="xml"))
        v = spec.validation_rules
        rules = [
            f"Allowed labels: {', '.join(v.get('allowed_labels', spec.label_names))}.",
            "Every <text> must be copied verbatim from the reviewer comment; paraphrased, merged or shortened spans are rejected by the validator.",
            "Empty spans and labels outside the allowed set are rejected.",
            "If nothing applies, return <annotations></annotations>.",
        ]
        sections.append(PromptSection("validation", "Validation rules applied to your output", "\n".join(f"- {r}" for r in rules), kind="list"))
        return PromptIR(sections=sections, spec_hash=spec.hash(), spec_kind=spec.kind)


class BackendRenderer:
    """``R_M``: PromptIR -> backend-specific system prompt string."""

    def render(self, ir: PromptIR, family: str) -> str:
        family = (family or "generic").lower()
        if family == "anthropic":
            parts = []
            for s in ir.sections:
                parts.append(f"<{s.section_id}>\n{s.body}\n</{s.section_id}>")
            parts.append("<response_rules>\nRespond with the XML only. Do not add explanations, headings or Markdown code fences.\n</response_rules>")
            return "\n\n".join(parts)
        if family == "openai":
            parts = [f"## {s.title}\n{s.body}" for s in ir.sections]
            parts.append("## Response rules\nOutput the XML document only. No prose, no Markdown code fences.")
            return "\n\n".join(parts)
        if family == "google":
            parts = [f"{i + 1}. {s.title}:\n{s.body}" for i, s in enumerate(ir.sections)]
            parts.append(f"{len(ir.sections) + 1}. Response rules:\nOutput only the XML document. Do not include any other text.")
            return "\n\n".join(parts)
        # minimax / generic
        parts = [f"### {s.title}\n{s.body}" for s in ir.sections]
        parts.append("### Response rules\nReturn only the XML document, without code fences or commentary.")
        return "\n\n".join(parts)


class PromptSynthesizer:
    """Optional LLM prompt synthesis with the Table 2 meta-prompt."""

    def __init__(self, backend: LLMBackend, meta_prompt_path: Optional[str] = None):
        self.backend = backend
        path = Path(meta_prompt_path) if meta_prompt_path else repo_root() / "prompts" / "actor_meta_prompt.txt"
        self.meta_prompt = path.read_text(encoding="utf-8")

    def synthesize(self, spec: TaskSpecification) -> tuple[str, dict]:
        spec_json = spec.to_json()
        demos = render_demonstrations(spec.demonstrations[:2])
        user = f"Task specification (JSON):\n{spec_json}\n\nSample annotated data:\n{demos or '(none)'}"
        resp = self.backend.complete([ChatMessage("system", self.meta_prompt), ChatMessage("user", user)], SamplingParams(temperature=0.0, max_tokens=2048), tag="prompt_synthesis")
        return resp.text.strip(), {"model_returned": resp.model_returned, "created_at": resp.created_at, "cached": resp.cached}


def load_fixed_instruction_baseline(path: Optional[str] = None) -> str:
    p = Path(path) if path else repo_root() / "prompts" / "fixed_instruction_baseline.txt"
    return p.read_text(encoding="utf-8").strip()


def build_prompt_bundle(
    spec: Optional[TaskSpecification],
    backend: LLMBackend,
    config: PipelineConfig,
    synthesizer: Optional[PromptSynthesizer] = None,
    fib_path: Optional[str] = None,
) -> PromptBundle:
    """Assemble the prompt used by the Analyzer for one (dataset, backend, configuration)."""
    if not config.use_adaptive_prompting:
        system = load_fixed_instruction_baseline(fib_path)
        return PromptBundle(
            system_prompt=system,
            user_template=FIB_USER_TEMPLATE,
            mode="fixed",
            family="none",
            spec_hash=spec.hash() if spec else None,
            prompt_hash=sha256_text(system)[:16],
            provenance={"source": "prompts/fixed_instruction_baseline.txt", "note": "dataset-independent; no examples, no dataset profile, no backend-specific rendering"},
        )
    if spec is None:
        raise ValueError("adaptive prompting requires a task specification")
    synthesized = None
    prov: dict = {"spec_kind": spec.kind, "renderer_family": backend.family, "synthesis": config.prompt_synthesis}
    if config.prompt_synthesis == "llm":
        synthesizer = synthesizer or PromptSynthesizer(backend)
        synthesized, meta = synthesizer.synthesize(spec)
        prov["synthesis_meta"] = meta
        prov["synthesized_text"] = synthesized
    ir = PromptCompiler().compile(spec, synthesized_guidelines=synthesized)
    system = BackendRenderer().render(ir, backend.family)
    return PromptBundle(
        system_prompt=system,
        user_template=USER_TEMPLATE,
        mode="adaptive",
        family=backend.family,
        spec_hash=spec.hash(),
        prompt_hash=sha256_text(system)[:16],
        provenance=prov,
    )
