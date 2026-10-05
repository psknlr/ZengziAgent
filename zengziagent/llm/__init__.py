"""LLM backends.

All experiments go through the OpenAI-compatible *chat completions* protocol, which is
offered by OpenRouter, Poe and MiniMax (and by OpenAI itself and Anthropic's
compatibility endpoint).  Every call is logged with the exact model identifier
returned by the provider, the sampling parameters, timestamps and token usage so
that the reproducibility subsection of the manuscript can be filled from the run
manifests rather than from memory.
"""
from .base import ChatMessage, LLMBackend, LLMResponse  # noqa: F401
from .cache import ResponseCache  # noqa: F401
from .mock import MockBackend  # noqa: F401
from .base import LLMRequestError  # noqa: F401
from .openai_compat import PROVIDERS, OpenAICompatibleBackend, available_aliases, default_backends, resolve_backend  # noqa: F401
