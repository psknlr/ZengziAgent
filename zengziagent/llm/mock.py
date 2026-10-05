"""Deterministic mock backend for pipeline tests and smoke runs.

The mock reads the reviewer comment from the ``<review>`` block of the user message,
segments it into sentences, and labels sentences with a small cue lexicon.  A seeded
noise model makes it emit the failure modes the Recorder and the Text Alignment Tool
are designed to catch (paraphrased spans, boundary shifts, malformed XML, illegal
labels).  **Mock outputs carry no scientific meaning**: they only exercise code paths.
"""
from __future__ import annotations

import random
import re
import time

from ..preprocessor import segment_sentences
from ..schema import SamplingParams
from ..utils import sha256_text, utc_now
from .base import ChatMessage, LLMBackend, LLMResponse

_REVIEW_RE = re.compile(r"<review[^>]*>(.*?)</review>", re.S)
_PREVIOUS_RE = re.compile(r"<previous_annotation>(.*?)</previous_annotation>", re.S)

_POS = ["well-written", "well written", "clear", "interesting", "novel", "strong", "convincing", "important", "solid", "elegant", "impressive", "thorough", "good", "nice", "well motivated", "well-motivated", "useful", "valuable", "significant", "sound"]
_NEG = ["insufficient", "unclear", "weak", "limited", "missing", "lack", "not clear", "concern", "problem", "poor", "confusing", "fails", "however", "unfortunately", "incorrect", "questionable", "difficult", "not convinced", "issue", "wonder whether"]
_JUS = ["because", "since", "for instance", "for example", "e.g.", "as shown", "given that", "due to", "this is", "in particular", "specifically", "such as", "the reason"]
_MC = ["overall", "in summary", "i recommend", "recommend", "accept", "reject", "to summarize", "in conclusion", "my recommendation"]

PROFILES = {
    "default": {"paraphrase": 0.15, "shift": 0.10, "drop": 0.10, "malformed": 0.08, "illegal": 0.04, "hallucinate": 0.05},
    "clean": {"paraphrase": 0.0, "shift": 0.0, "drop": 0.0, "malformed": 0.0, "illegal": 0.0, "hallucinate": 0.0},
    "noisy": {"paraphrase": 0.35, "shift": 0.25, "drop": 0.2, "malformed": 0.2, "illegal": 0.1, "hallucinate": 0.15},
}


class MockBackend(LLMBackend):
    provider = "mock"
    family = "generic"

    def __init__(self, profile: str = "default", seed: int = 0):
        self.model = f"mock-{profile}"
        self.noise = PROFILES.get(profile, PROFILES["default"])
        self.seed = seed

    def complete(self, messages: list[ChatMessage], params: SamplingParams | None = None, run_index: int = 0, tag: str = "") -> LLMResponse:
        t0 = time.time()
        users = [m.content for m in messages if m.role == "user"]
        user = users[-1] if users else ""
        system = next((m.content for m in messages if m.role == "system"), "")
        if "prompt engineer" in system.lower():  # Table 2 meta-prompt (prompt synthesis)
            text = self._synthesize_prompt(user)
        else:
            # the reviewer comment is in the first user message; later user turns carry
            # validation feedback (refinement) - exactly what a real backend sees
            m = None
            for u in users:
                m = _REVIEW_RE.search(u)
                if m:
                    break
            review = m.group(1).strip() if m else user
            retry = any("failed validation" in u for u in users)
            reflection = any(_PREVIOUS_RE.search(u) is not None for u in users)
            rng = random.Random(f"{self.seed}|{run_index}|{sha256_text(review)}|{retry}|{reflection}")
            text = self._annotate(review, rng, retry=retry, reflection=reflection)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model_requested=self.model,
            model_returned=self.model,
            finish_reason="stop",
            prompt_tokens=len(user.split()),
            completion_tokens=len(text.split()),
            latency_s=round(time.time() - t0, 4),
            created_at=utc_now(),
            params=(params or SamplingParams()).to_dict(),
        )

    # --------------------------------------------------------------- internals
    def _synthesize_prompt(self, user: str) -> str:
        return (
            "Task description: annotate reviewer comments with Eval_pos, Eval_neg, Jus_pos, Jus_neg and Major_Claim spans.\n"
            "Guidelines: copy spans verbatim; use only permitted labels; one annotation per XML block; "
            "a justification explains why an evaluation holds; Major_Claim is the overall recommendation.\n"
        )

    def _annotate(self, review: str, rng: random.Random, retry: bool, reflection: bool) -> str:
        noise = dict(self.noise)
        if retry or reflection:
            noise = {k: v * 0.25 for k, v in noise.items()}
            noise["malformed"] = 0.0
            noise["illegal"] = 0.0
        blocks = []
        prev_label = None
        for s, e in segment_sentences(review):
            sent = review[s:e]
            low = sent.lower()
            label = None
            if any(c in low for c in _MC):
                label = "Major_Claim"
            elif any(c in low for c in _JUS) and prev_label in {"Eval_pos", "Eval_neg"}:
                label = "Jus_pos" if prev_label == "Eval_pos" else "Jus_neg"
            elif any(c in low for c in _NEG):
                label = "Eval_neg"
            elif any(c in low for c in _POS):
                label = "Eval_pos"
            prev_label = label
            if label is None or len(sent) < 12:
                continue
            if rng.random() < noise["drop"]:
                continue
            span = sent
            r = rng.random()
            if r < noise["paraphrase"]:
                span = _paraphrase(span)
            elif r < noise["paraphrase"] + noise["shift"]:
                words = span.split()
                if len(words) > 4:
                    span = " ".join(words[1:-1])
            if rng.random() < noise["illegal"]:
                label = "Eval_neutral"
            blocks.append(f"  <annotation>\n    <text>{span}</text>\n    <index>{s}-{e}</index>\n    <label>{label}</label>\n  </annotation>")
        if rng.random() < noise["hallucinate"]:
            blocks.append("  <annotation>\n    <text>The reviewers unanimously praise the extensive supplementary appendix.</text>\n    <label>Eval_pos</label>\n  </annotation>")
        body = "\n".join(blocks)
        if rng.random() < noise["malformed"]:
            return f"<annotations>\n{body}\n"  # missing closing root tag
        return f"<annotations>\n{body}\n</annotations>"


def _paraphrase(span: str) -> str:
    out = span.replace("insufficient", "not sufficient").replace("unclear", "not clear").replace("well-written", "well written")
    if out == span:
        out = span.rstrip(".") + " here."
    return out
