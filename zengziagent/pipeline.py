"""Algorithm 1 - the ZengziAgent annotation pipeline with ablation switches."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .actor import PromptBundle
from .alignment import TextAligner, exact_only_aligner
from .analyzer import Analyzer, ParsedOutput
from .llm.base import LLMBackend, LLMRequestError, LLMResponse
from .preprocessor import Preprocessor
from .recorder import AlignedSpan, Recorder, ValidationReport, Validator
from .schema import PipelineConfig, ReviewRecord, SamplingParams
from .tokenization import get_tokenizer


@dataclass
class Attempt:
    kind: str  # initial | refinement | reflection
    raw_output: str
    n_parsed: int
    validation: dict
    latency_s: float
    model_returned: Optional[str]
    cached: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ReviewResult:
    review_id: str
    dataset: str
    config_id: str
    backend: str
    model: str
    run_index: int
    round: int  # 1 = R1, 2 = R2 (reflection)
    attempts: list[Attempt] = field(default_factory=list)
    spans: list[AlignedSpan] = field(default_factory=list)
    retries: int = 0
    flagged: bool = False
    final_validation: Optional[dict] = None
    prompt_hash: str = ""
    spec_hash: Optional[str] = None
    preprocessing: str = ""
    alignment: dict = field(default_factory=dict)
    processed_text_len: int = 0
    timing_s: float = 0.0
    error: Optional[str] = None
    final_xml: str = ""

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "dataset": self.dataset,
            "config_id": self.config_id,
            "backend": self.backend,
            "model": self.model,
            "run_index": self.run_index,
            "round": self.round,
            "retries": self.retries,
            "flagged": self.flagged,
            "prompt_hash": self.prompt_hash,
            "spec_hash": self.spec_hash,
            "preprocessing": self.preprocessing,
            "alignment": self.alignment,
            "processed_text_len": self.processed_text_len,
            "timing_s": round(self.timing_s, 3),
            "error": self.error,
            "final_validation": self.final_validation,
            "attempts": [a.to_dict() for a in self.attempts],
            "spans": [s.to_dict() for s in self.spans],
            "final_xml": self.final_xml,
        }


class ZengziAgentPipeline:
    def __init__(
        self,
        backend: LLMBackend,
        config: PipelineConfig,
        bundle: PromptBundle,
        allowed_labels: list[str],
        sampling: SamplingParams | None = None,
        tokenizer=None,
        reflection_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.config = config
        self.bundle = bundle
        tok = tokenizer or get_tokenizer("auto")
        self.preprocessor = Preprocessor(enabled=config.use_preprocessing)
        if config.use_alignment:
            self.aligner = TextAligner(tokenizer=tok, similarity_threshold=config.alignment_similarity_threshold)
        else:
            self.aligner = exact_only_aligner(tokenizer=tok)
        self.validator = Validator(
            allowed_labels=allowed_labels,
            aligner=self.aligner,
            validate_structure=config.use_validation,
            validate_labels=config.use_validation,
            validate_alignment=config.use_validation,
        )
        self.recorder = Recorder(keep_illegal_labels=not config.use_validation)
        self.analyzer = Analyzer(backend, sampling, reflection_prompt=reflection_prompt)

    # ---------------------------------------------------------------- Algorithm 1
    def run(self, review: ReviewRecord, run_index: int = 0, previous_xml: Optional[str] = None) -> ReviewResult:
        t0 = time.time()
        result = ReviewResult(
            review_id=review.review_id,
            dataset=review.dataset,
            config_id=self.config.config_id,
            backend=self.backend.label,
            model=self.backend.model,
            run_index=run_index,
            round=2 if previous_xml is not None else 1,
            prompt_hash=self.bundle.prompt_hash,
            spec_hash=self.bundle.spec_hash,
            preprocessing=self.preprocessor.name,
            alignment=self.aligner.describe(),
        )
        try:
            processed = self.preprocessor.process(review.text)  # line 3
            result.processed_text_len = len(processed.text)
            feedback: Optional[str] = None
            previous_output: Optional[str] = None
            spans: list[AlignedSpan] = []
            report: Optional[ValidationReport] = None
            attempts_allowed = 1 + (self.config.max_refinement_retries if (self.config.use_validation and self.config.use_refinement) else 0)
            for attempt in range(attempts_allowed):
                kind = "reflection" if previous_xml is not None and attempt == 0 else ("refinement" if attempt > 0 else "initial")
                resp, parsed = self.analyzer.annotate(  # line 4 (and 6)
                    self.bundle,
                    review.review_id,
                    processed.text,
                    dataset=review.dataset,
                    run_index=run_index,
                    validation_feedback=feedback,
                    previous_xml=previous_xml if attempt == 0 else None,
                    previous_output=previous_output,
                )
                report, spans = self.validator.validate(parsed, processed)  # line 5 + 7
                result.attempts.append(
                    Attempt(kind, resp.text, len(parsed.annotations), report.to_dict(), resp.latency_s, resp.model_returned, resp.cached, resp.prompt_tokens, resp.completion_tokens, resp.finish_reason)
                )
                result.final_xml = resp.text
                if not (self.config.use_validation and report.blocking):
                    break
                if attempt + 1 >= attempts_allowed:
                    break
                feedback = report.feedback()
                previous_output = resp.text
                result.retries += 1
                if previous_xml is not None:
                    previous_xml = None  # refinement of a reflection output proceeds as a normal refinement
            result.spans = self.recorder.finalize(spans)  # line 8-9
            result.final_validation = report.to_dict() if report else None
            result.flagged = bool(report and report.blocking)
        except LLMRequestError:
            raise  # authentication / unknown model: abort the run instead of persisting empty results
        except Exception as exc:  # keep the run going; the evaluator treats errors as empty output
            result.error = f"{type(exc).__name__}: {exc}"
        result.timing_s = time.time() - t0
        return result
