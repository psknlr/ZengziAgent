"""OpenAI-compatible chat-completions client for OpenRouter, Poe, MiniMax (and others).

Provider presets::

    openrouter   https://openrouter.ai/api/v1      OPENROUTER_API_KEY
    poe          https://api.poe.com/v1            POE_API_KEY
    minimax      https://api.minimax.io/v1         MINIMAX_API_KEY   (international)
    minimax-cn   https://api.minimaxi.com/v1       MINIMAX_API_KEY   (mainland China)
    openai       https://api.openai.com/v1         OPENAI_API_KEY
    anthropic    https://api.anthropic.com/v1      ANTHROPIC_API_KEY (OpenAI-compatible endpoint)
    custom       $LLM_BASE_URL                     LLM_API_KEY

Backend aliases (``claude``, ``gpt4o``, ``gemini``, ``minimax``) are resolved through
``configs/backends.yaml``; a raw model identifier can always be given explicitly as
``provider:model=<id>``.  The identifier reported by the provider in each response is
recorded next to the requested one so that rolling aliases (e.g. ``chatgpt-4o-latest``)
are resolved to the snapshot that actually served the run.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from ..schema import SamplingParams
from ..utils import LOG, utc_now
from .base import ChatMessage, LLMBackend, LLMRequestError, LLMResponse, infer_family
from .cache import ResponseCache


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str
    extra_headers: dict
    supports_seed: bool = True


PROVIDERS: dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        {"HTTP-Referer": "https://github.com/psknlr/ZengziAgent", "X-Title": "ZengziAgent"},
    ),
    "poe": ProviderSpec("poe", "https://api.poe.com/v1", "POE_API_KEY", {}, supports_seed=False),  # Poe documents 'seed' as ignored
    "minimax": ProviderSpec("minimax", "https://api.minimax.io/v1", "MINIMAX_API_KEY", {}, supports_seed=False),
    "minimax-cn": ProviderSpec("minimax-cn", "https://api.minimaxi.com/v1", "MINIMAX_API_KEY", {}, supports_seed=False),
    "openai": ProviderSpec("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", {}),
    "anthropic": ProviderSpec("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", {}, supports_seed=False),
    "custom": ProviderSpec("custom", os.environ.get("LLM_BASE_URL", ""), "LLM_API_KEY", {}),
}

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}




class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        family: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        timeout: float = 180.0,
        max_retries: int = 6,
        extra_headers: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        min_interval_s: float = 0.0,
    ):
        import httpx  # noqa: WPS433

        spec = PROVIDERS.get(provider)
        if spec is None and base_url is None:
            raise ValueError(f"unknown provider '{provider}'; known: {sorted(PROVIDERS)} or pass base_url")
        self.provider = provider
        self.model = model
        self.family = family or infer_family(model)
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.api_key = api_key or (os.environ.get(spec.api_key_env) if spec else None) or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            env = spec.api_key_env if spec else "LLM_API_KEY"
            raise RuntimeError(f"No API key for provider '{provider}': set ${env}")
        self.supports_seed = spec.supports_seed if spec else True
        self.cache = cache
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        headers.update(spec.extra_headers if spec else {})
        headers.update(extra_headers or {})
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    # ----------------------------------------------------------------- API
    def list_models(self) -> list[str]:
        try:
            r = self._client.get("/models")
            r.raise_for_status()
            data = r.json().get("data", [])
            return [m.get("id") for m in data if isinstance(m, dict)]
        except Exception as exc:  # pragma: no cover - network
            LOG.warning("could not list models from %s: %s", self.provider, exc)
            return []

    def check_model_available(self) -> Optional[bool]:
        ids = self.list_models()
        if not ids:
            return None
        return self.model in ids

    def probe_capabilities(self) -> dict:
        """Look the model up in the provider catalogue (once) and record what it supports.

        OpenRouter exposes ``supported_parameters`` per model; ``seed`` is only sent when the
        catalogue says the model honours it, so the manifest never claims a seed that was ignored."""
        if getattr(self, "_capabilities", None) is not None:
            return self._capabilities
        caps: dict = {"catalogue_checked": False, "listed": None, "supported_parameters": None}
        try:
            r = self._client.get("/models")
            r.raise_for_status()
            entries = [m for m in r.json().get("data", []) if isinstance(m, dict)]
            caps["catalogue_checked"] = True
            entry = next((m for m in entries if m.get("id") == self.model), None)
            caps["listed"] = entry is not None
            if entry is not None:
                sp = entry.get("supported_parameters")
                if isinstance(sp, list):
                    caps["supported_parameters"] = sp
                    self.supports_seed = self.supports_seed and ("seed" in sp)
                cw = entry.get("context_length") or (entry.get("context_window") or {}).get("context_length")
                if cw:
                    caps["context_length"] = cw
                top = entry.get("top_provider") or {}
                if top.get("max_completion_tokens"):
                    caps["max_completion_tokens"] = top["max_completion_tokens"]
        except Exception as exc:  # pragma: no cover - network
            LOG.warning("capability probe failed for %s: %s", self.provider, exc)
        caps["seed_sent"] = bool(self.supports_seed)
        self._capabilities = caps
        return caps

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "family": self.family, "base_url": self.base_url, "supports_seed": self.supports_seed}

    def complete(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
        run_index: int = 0,
        tag: str = "",
    ) -> LLMResponse:
        params = params or SamplingParams()
        msg_dicts = [m.to_dict() for m in messages]
        body = {
            "model": self.model,
            "messages": msg_dicts,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None and self.supports_seed:
            body["seed"] = params.seed + run_index
        body.update(self.extra_body)
        key = ResponseCache.make_key(self.provider, self.model, msg_dicts, {k: v for k, v in body.items() if k != "messages"}, run_index)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                resp = LLMResponse(**hit)
                resp.cached = True
                return resp

        t0 = time.time()
        raw = self._post_with_retry("/chat/completions", body)
        latency = time.time() - t0
        choice = raw["choices"][0]
        message = choice.get("message") or {}
        text = _extract_text(message)
        usage = raw.get("usage") or {}
        resp = LLMResponse(
            text=text,
            provider=self.provider,
            model_requested=self.model,
            model_returned=raw.get("model"),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=round(latency, 3),
            created_at=utc_now(),
            request_id=raw.get("id"),
            cached=False,
            params={k: v for k, v in body.items() if k != "messages"},
        )
        cacheable = bool(text.strip()) and (resp.finish_reason or "stop") not in _NON_CACHEABLE_FINISH
        if self.cache is not None and cacheable:
            self.cache.put(key, self.provider, self.model, run_index, tag, body, resp.to_dict())
        elif not cacheable:
            LOG.warning("%s response for %s not cached (finish_reason=%s, empty=%s)", self.provider, tag, resp.finish_reason, not text.strip())
        return resp

    # ------------------------------------------------------------ internals
    def _post_with_retry(self, path: str, body: dict) -> dict:
        import httpx  # noqa: WPS433

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if self.min_interval_s:
                wait = self.min_interval_s - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
            try:
                self._last_call = time.time()
                r = self._client.post(path, json=body)
                if r.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError(f"retryable status {r.status_code}: {r.text[:300]}", request=r.request, response=r)
                if r.status_code >= 400:  # 4xx other than rate limits: retrying cannot help
                    raise LLMRequestError(f"{self.provider} HTTP {r.status_code} for model '{self.model}': {r.text[:500]}")
                data = r.json()
                _validate_payload(data)
                return data
            except LLMRequestError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                backoff = min(60.0, (2**attempt) + random.uniform(0, 1))
                LOG.warning("%s call failed (attempt %d/%d): %s; retrying in %.1fs", self.provider, attempt + 1, self.max_retries, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_err}")


_NON_CACHEABLE_FINISH = {"length", "content_filter", "error"}


def _extract_text(message: dict) -> str:
    text = message.get("content")
    if isinstance(text, list):  # content parts
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return text or ""


def _validate_payload(data: dict) -> None:
    """Raise a *retryable* error for HTTP-200 payloads that carry no usable completion."""
    if not isinstance(data, dict):
        raise RuntimeError("provider returned a non-object payload")
    if "error" in data and not data.get("choices"):
        raise RuntimeError(f"provider error: {str(data['error'])[:300]}")
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"provider returned no choices: {str(data)[:300]}")
    choice = choices[0]
    if choice.get("error"):
        raise RuntimeError(f"choice-level error: {str(choice['error'])[:300]}")
    if choice.get("finish_reason") == "error":
        raise RuntimeError("provider reported finish_reason=error")
    if not _extract_text(choice.get("message") or {}).strip():
        raise RuntimeError(f"empty completion (finish_reason={choice.get('finish_reason')})")


def resolve_backend(
    spec: str,
    backends_yaml: Optional[str] = None,
    cache: Optional[ResponseCache] = None,
    **kwargs,
) -> LLMBackend:
    """Resolve ``provider:alias`` / ``provider:model=<id>`` / ``mock`` into a backend object.

    Examples: ``openrouter:claude``, ``poe:gpt4o``, ``minimax:minimax``,
    ``openrouter:model=anthropic/claude-3.5-sonnet``, ``mock``, ``mock:noisy``.
    """
    from .mock import MockBackend

    if spec == "mock" or spec.startswith("mock:"):
        profile = spec.split(":", 1)[1] if ":" in spec else "default"
        return MockBackend(profile=profile)
    if ":" not in spec:
        raise ValueError("backend spec must be 'provider:alias', 'provider:model=<id>' or 'mock'")
    provider, rest = spec.split(":", 1)
    if rest.startswith("model="):
        model = rest[len("model=") :]
        family = None
    else:
        import yaml  # noqa: WPS433

        from ..utils import repo_root

        path = backends_yaml or str(repo_root() / "configs" / "backends.yaml")
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        entry = cfg.get("backends", {}).get(rest)
        if entry is None:
            raise ValueError(f"unknown backend alias '{rest}' in {path}; known: {sorted(cfg.get('backends', {}))}")
        model = entry.get("models", {}).get(provider)
        if model is None:
            raise ValueError(f"alias '{rest}' has no model configured for provider '{provider}' in {path}")
        family = entry.get("family")
        prov_cfg = cfg.get("providers", {}).get(provider, {})
        if prov_cfg.get("base_url"):
            kwargs.setdefault("base_url", prov_cfg["base_url"])
    return OpenAICompatibleBackend(provider=provider, model=model, family=family, cache=cache, **kwargs)
