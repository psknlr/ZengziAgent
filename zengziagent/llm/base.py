from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..schema import SamplingParams


class LLMRequestError(RuntimeError):
    """Non-retryable provider error (authentication, unknown model, malformed request ...).

    Raised through the pipeline so that a run aborts instead of persisting empty results."""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model_requested: str
    model_returned: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_s: float = 0.0
    created_at: str = ""
    request_id: Optional[str] = None
    cached: bool = False
    params: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class LLMBackend(abc.ABC):
    """Minimal backend interface used by the Actor (prompt synthesis) and Analyzer."""

    provider: str = "abstract"
    model: str = "abstract"
    family: str = "generic"  # anthropic | openai | google | minimax | generic  (drives R_M rendering)

    @abc.abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
        run_index: int = 0,
        tag: str = "",
    ) -> LLMResponse:  # pragma: no cover - interface
        ...

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "family": self.family}

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def infer_family(model_id: str) -> str:
    m = model_id.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m or m.startswith("o1") or m.startswith("o3") or "chatgpt" in m:
        return "openai"
    if "gemini" in m or "google" in m:
        return "google"
    if "minimax" in m or "abab" in m:
        return "minimax"
    return "generic"
