# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""LLM provider adapters for model listing and chat completions."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from .ai_store import ModelProfile
from .config import APP_NAME, Settings
from .job_cancel import JobCancelled, ShouldCancel

PROVIDER_LABELS: dict[str, str] = {
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google (Gemini)",
    "grok": "Grok (xAI)",
    "ollama_cloud": "Ollama Cloud",
    "ollama_local": "Ollama (Local)",
}


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    key_field: str | None
    needs_key: bool


PROVIDERS: dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec("openrouter", "OpenRouter", "ai_key_openrouter", True),
    "openai": ProviderSpec("openai", "OpenAI", "ai_key_openai", True),
    "anthropic": ProviderSpec("anthropic", "Anthropic", "ai_key_anthropic", True),
    "google": ProviderSpec("google", "Google (Gemini)", "ai_key_google", True),
    "grok": ProviderSpec("grok", "Grok (xAI)", "ai_key_grok", True),
    "ollama_cloud": ProviderSpec("ollama_cloud", "Ollama Cloud", "ai_key_ollama_cloud", True),
    "ollama_local": ProviderSpec("ollama_local", "Ollama (Local)", None, False),
}


#: Ollama sizes its context window per request and defaults to about 4k tokens,
#: silently dropping whatever does not fit rather than refusing the request. A
#: cleanup batch is far larger than that, so the model would see a fragment,
#: answer for a fragment, and the caller would read the short answer as a batch
#: that needs splitting — halving forever, because the batch was never the
#: problem. So num_ctx is always sent, sized to the prompt actually being made.
OLLAMA_CHARS_PER_TOKEN = 3.5
OLLAMA_MIN_NUM_CTX = 4_096
#: Every token of context costs KV-cache memory on the machine the user is
#: sitting at. 32k holds a large cleanup batch and still loads on a laptop.
OLLAMA_MAX_NUM_CTX = 32_768
OLLAMA_CTX_HEADROOM_TOKENS = 512


def ollama_num_ctx(system_prompt: str, user_prompt: str, max_output_tokens: int) -> int:
    """A context window big enough for this prompt and its answer.

    Rounded up to a power of two because that is how these runtimes like their
    cache shapes, and clamped so one huge batch cannot try to allocate a window
    the machine has no memory for.
    """
    prompt_tokens = (len(system_prompt) + len(user_prompt)) / OLLAMA_CHARS_PER_TOKEN
    needed = int(prompt_tokens + max(0, max_output_tokens) + OLLAMA_CTX_HEADROOM_TOKENS)
    size = OLLAMA_MIN_NUM_CTX
    while size < needed and size < OLLAMA_MAX_NUM_CTX:
        size *= 2
    return min(size, OLLAMA_MAX_NUM_CTX)


class ProviderError(RuntimeError):
    pass


EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral"}

# explicit = cache_control breakpoints; prefix = stable prefix + optional cache key;
# implicit = stable prefix ordering only (provider may auto-cache).
PROMPT_CACHE_MODES: dict[str, str] = {
    "anthropic": "explicit",
    "openai": "prefix",
    "openrouter": "prefix",
    "google": "implicit",
    "grok": "implicit",
}


def prompt_cache_mode(provider: str) -> str | None:
    return PROMPT_CACHE_MODES.get(provider)


def format_prompt_cache_usage(usage: dict[str, int]) -> str | None:
    read = int(usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0)
    create = int(usage.get("cache_creation_input_tokens") or 0)
    if not read and not create:
        return None
    parts: list[str] = []
    if read:
        parts.append(f"read {read:,}")
    if create:
        parts.append(f"wrote {create:,}")
    return f"prompt cache — {', '.join(parts)} tok"


def _key(settings: Settings, provider: str) -> str:
    spec = PROVIDERS[provider]
    if not spec.key_field:
        return ""
    return getattr(settings, spec.key_field, "").strip()


def is_provider_configured(settings: Settings, provider: str) -> bool:
    if provider not in PROVIDERS:
        return False
    if provider == "ollama_local":
        return bool((settings.ollama_local_url or "").strip())
    spec = PROVIDERS[provider]
    if not spec.needs_key:
        return True
    return bool(_key(settings, provider))


def configured_providers(settings: Settings) -> list[str]:
    return [pid for pid in PROVIDERS if is_provider_configured(settings, pid)]


def list_models(settings: Settings, provider: str) -> list[str]:
    if not is_provider_configured(settings, provider):
        return []
    fetchers: dict[str, Callable[[Settings], list[str]]] = {
        "openrouter": _list_openrouter,
        "openai": _list_openai,
        "anthropic": _list_anthropic,
        "google": _list_google,
        "grok": _list_grok,
        "ollama_cloud": _list_ollama_cloud,
        "ollama_local": _list_ollama_local,
    }
    try:
        models = fetchers[provider](settings)
        return sorted(set(models), key=str.lower)
    except Exception as e:
        raise ProviderError(f"Could not list models for {PROVIDER_LABELS.get(provider, provider)}: {e}") from e


def test_provider(settings: Settings, provider: str) -> str:
    """Verify provider credentials by listing models. Returns a short success message."""
    label = PROVIDER_LABELS.get(provider, provider)
    if provider == "ollama_local":
        if not (settings.ollama_local_url or "").strip():
            raise ProviderError("Enter a URL for Ollama local.")
    elif PROVIDERS[provider].needs_key and not _key(settings, provider):
        raise ProviderError("Enter an API key first.")
    models = list_models(settings, provider)
    if not models:
        return f"{label}: connected, but no models were returned."
    sample = ", ".join(models[:3])
    suffix = f" (+{len(models) - 3} more)" if len(models) > 3 else ""
    return f"{label}: OK — {len(models)} model(s), e.g. {sample}{suffix}"


def chat_completion(
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    *,
    timeout: float | tuple[float, float] = 180,
    stream: bool = False,
    should_cancel: ShouldCancel = None,
    stream_cb: Callable[[int], None] | None = None,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = True,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    cache_enabled = use_prompt_cache and prompt_cache_mode(provider) is not None
    common = dict(
        settings=settings,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        profile=profile,
        timeout=timeout,
        cacheable_user_prefix=cacheable_user_prefix,
        cache_key=cache_key,
        use_prompt_cache=cache_enabled,
        usage_cb=usage_cb,
    )
    if provider == "anthropic" and stream:
        return _chat_anthropic(
            **common,
            stream=True,
            should_cancel=should_cancel,
            stream_cb=stream_cb,
        )
    callers: dict[str, Callable[..., str]] = {
        "openrouter": _chat_openai_compat,
        "openai": _chat_openai_compat,
        "grok": _chat_openai_compat,
        "ollama_cloud": _chat_ollama_cloud,
        "ollama_local": _chat_ollama_local,
        "anthropic": _chat_anthropic,
        "google": _chat_google,
    }
    return callers[provider](**common)


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

def _list_openrouter(settings: Settings) -> list[str]:
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {_key(settings, 'openrouter')}"},
        timeout=45,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if m.get("id")]


def _list_openai(settings: Settings) -> list[str]:
    r = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {_key(settings, 'openai')}"},
        timeout=45,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if m.get("id")]


def _list_anthropic(settings: Settings) -> list[str]:
    r = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": _key(settings, "anthropic"),
            "anthropic-version": "2023-06-01",
        },
        timeout=45,
    )
    r.raise_for_status()
    data = r.json().get("data", r.json().get("models", []))
    return [m["id"] for m in data if m.get("id")]


def _list_google(settings: Settings) -> list[str]:
    r = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": _key(settings, "google")},
        timeout=45,
    )
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        name = m.get("name", "")
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name and "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(name)
    return out


def _list_grok(settings: Settings) -> list[str]:
    r = requests.get(
        "https://api.x.ai/v1/models",
        headers={"Authorization": f"Bearer {_key(settings, 'grok')}"},
        timeout=45,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if m.get("id")]


def _list_ollama_cloud(settings: Settings) -> list[str]:
    r = requests.get(
        "https://ollama.com/api/tags",
        headers={"Authorization": f"Bearer {_key(settings, 'ollama_cloud')}"},
        timeout=45,
    )
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", []) if m.get("name")]


def _list_ollama_local(settings: Settings) -> list[str]:
    base = (settings.ollama_local_url or "http://localhost:11434").rstrip("/")
    r = requests.get(f"{base}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", []) if m.get("name")]


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

def _openai_compat_url(provider: str) -> str:
    return {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "grok": "https://api.x.ai/v1/chat/completions",
    }[provider]


def _openai_compat_headers(settings: Settings, provider: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider == "openrouter":
        headers["Authorization"] = f"Bearer {_key(settings, 'openrouter')}"
        headers["HTTP-Referer"] = "https://github.com/plaud-whisper-studio"
        headers["X-Title"] = APP_NAME
    elif provider == "openai":
        headers["Authorization"] = f"Bearer {_key(settings, 'openai')}"
    elif provider == "grok":
        headers["Authorization"] = f"Bearer {_key(settings, 'grok')}"
    return headers


def _normalize_cache_key(cache_key: str) -> str:
    return cache_key.strip()[:128]


def _emit_usage(usage_cb: Callable[[dict[str, int]], None] | None, usage: dict[str, int]) -> None:
    if usage_cb and usage:
        usage_cb(usage)


def _anthropic_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage") or {}
    return {
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        "input_tokens": int(usage.get("input_tokens") or 0),
    }


def _openai_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
    }


def _google_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usageMetadata") or {}
    return {
        "cached_tokens": int(usage.get("cachedContentTokenCount") or 0),
        "input_tokens": int(usage.get("promptTokenCount") or 0),
    }


def _combined_user_prompt(cacheable_user_prefix: str | None, user_prompt: str) -> str:
    if cacheable_user_prefix:
        return f"{cacheable_user_prefix}{user_prompt}"
    return user_prompt


def _chat_openai_compat(
    *,
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    timeout: float | tuple[float, float] = 180,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = False,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _combined_user_prompt(cacheable_user_prefix, user_prompt)},
        ],
    }
    if use_prompt_cache and cache_key and provider in ("openai", "openrouter"):
        body["prompt_cache_key"] = _normalize_cache_key(cache_key)
    if profile.use_max_completion_tokens:
        body["max_completion_tokens"] = profile.max_tokens
    else:
        body["max_tokens"] = profile.max_tokens
    if not profile.omit_temperature:
        body["temperature"] = profile.temperature
    if provider in ("openai", "openrouter") and not profile.omit_json_response_format:
        body["response_format"] = {"type": "json_object"}

    r = requests.post(
        _openai_compat_url(provider),
        headers=_openai_compat_headers(settings, provider),
        json=body,
        timeout=timeout,
    )
    if not r.ok:
        raise ProviderError(_http_error_text(r))
    data = r.json()
    _emit_usage(usage_cb, _openai_usage(data))
    return _extract_openai_message_content(data)


def _anthropic_headers(settings: Settings) -> dict[str, str]:
    return {
        "x-api-key": _key(settings, "anthropic"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _anthropic_body(
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    *,
    stream: bool,
    cacheable_user_prefix: str | None = None,
    use_prompt_cache: bool = False,
) -> dict[str, Any]:
    cache = EPHEMERAL_CACHE if use_prompt_cache else None
    if cache and system_prompt:
        system: str | list[dict[str, Any]] = [
            {"type": "text", "text": system_prompt, "cache_control": cache},
        ]
    else:
        system = system_prompt

    if cache and cacheable_user_prefix:
        user_content: str | list[dict[str, Any]] = [
            {"type": "text", "text": cacheable_user_prefix, "cache_control": cache},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = _combined_user_prompt(cacheable_user_prefix, user_prompt)

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": profile.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    if stream:
        body["stream"] = True
    if not profile.omit_temperature:
        body["temperature"] = profile.temperature
    return body


def _request_timeouts(timeout: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(timeout, tuple):
        return timeout
    return 30.0, float(timeout)


def _chat_anthropic(
    *,
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    timeout: float | tuple[float, float] = 180,
    stream: bool = False,
    should_cancel: ShouldCancel = None,
    stream_cb: Callable[[int], None] | None = None,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = False,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    del provider, cache_key
    headers = _anthropic_headers(settings)
    body = _anthropic_body(
        model,
        system_prompt,
        user_prompt,
        profile,
        stream=stream,
        cacheable_user_prefix=cacheable_user_prefix,
        use_prompt_cache=use_prompt_cache,
    )
    connect_timeout, read_timeout = _request_timeouts(timeout)

    if not stream:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=(connect_timeout, read_timeout),
        )
        if not r.ok:
            raise ProviderError(_http_error_text(r))
        data = r.json()
        _emit_usage(usage_cb, _anthropic_usage(data))
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        if not text:
            stop = data.get("stop_reason") or ""
            hint = f" (stop_reason={stop})" if stop else ""
            raise ProviderError(f"Model returned an empty response{hint}.")
        return text

    headers["accept"] = "text/event-stream"
    text_parts: list[str] = []
    stop_reason = ""
    received = 0
    stream_usage: dict[str, int] = {}
    with requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    ) as response:
        if not response.ok:
            raise ProviderError(_http_error_text(response))
        for raw_line in response.iter_lines(decode_unicode=True):
            if should_cancel and should_cancel():
                response.close()
                raise JobCancelled("AI Cleanup: cancelled.")
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = raw_line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message") or {}
                stream_usage = _anthropic_usage({"usage": message.get("usage") or {}})
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    chunk = delta.get("text") or ""
                    if chunk:
                        text_parts.append(chunk)
                        received += len(chunk)
                        if stream_cb:
                            stream_cb(received)
            elif etype == "message_delta":
                delta = event.get("delta") or {}
                stop_reason = delta.get("stop_reason") or stop_reason
                usage = event.get("usage") or {}
                if usage:
                    stream_usage = _anthropic_usage({"usage": usage})
            elif etype == "error":
                err = event.get("error") or {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                raise ProviderError(message or str(event))

    text = "".join(text_parts).strip()
    _emit_usage(usage_cb, stream_usage)
    if not text:
        hint = f" (stop_reason={stop_reason})" if stop_reason else ""
        raise ProviderError(f"Model returned an empty response{hint}.")
    return text


def _chat_google(
    *,
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    timeout: float | tuple[float, float] = 180,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = False,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    del provider, cache_key
    model_name = model if model.startswith("models/") else f"models/{model}"
    user_parts: list[dict[str, str]] = []
    if cacheable_user_prefix:
        user_parts.append({"text": cacheable_user_prefix})
    user_parts.append({"text": user_prompt})
    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": {
            "maxOutputTokens": profile.max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if not profile.omit_temperature:
        body["generationConfig"]["temperature"] = profile.temperature
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent",
        params={"key": _key(settings, "google")},
        json=body,
        timeout=timeout,
    )
    if not r.ok:
        raise ProviderError(_http_error_text(r))
    data = r.json()
    _emit_usage(usage_cb, _google_usage(data))
    candidates = data.get("candidates", [])
    if not candidates:
        raise ProviderError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise ProviderError("Gemini returned an empty response.")
    return text


def _chat_ollama_cloud(
    *,
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    timeout: float | tuple[float, float] = 180,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = False,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    del cache_key, use_prompt_cache, usage_cb
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _combined_user_prompt(cacheable_user_prefix, user_prompt)},
        ],
        "stream": False,
        "format": "json",
    }
    options: dict[str, Any] = {
        "num_predict": profile.max_tokens,
        "num_ctx": ollama_num_ctx(
            system_prompt,
            _combined_user_prompt(cacheable_user_prefix, user_prompt),
            profile.max_tokens,
        ),
    }
    if not profile.omit_temperature:
        options["temperature"] = profile.temperature
    body["options"] = options
    r = requests.post(
        "https://ollama.com/api/chat",
        headers={"Authorization": f"Bearer {_key(settings, 'ollama_cloud')}"},
        json=body,
        timeout=timeout,
    )
    if not r.ok:
        raise ProviderError(_http_error_text(r))
    text = (r.json().get("message", {}) or {}).get("content", "") or ""
    if not str(text).strip():
        raise ProviderError("Ollama returned an empty response.")
    return str(text)


def _chat_ollama_local(
    *,
    settings: Settings,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    profile: ModelProfile,
    timeout: float | tuple[float, float] = 180,
    cacheable_user_prefix: str | None = None,
    cache_key: str | None = None,
    use_prompt_cache: bool = False,
    usage_cb: Callable[[dict[str, int]], None] | None = None,
) -> str:
    del cache_key, use_prompt_cache, usage_cb
    base = (settings.ollama_local_url or "http://localhost:11434").rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _combined_user_prompt(cacheable_user_prefix, user_prompt)},
        ],
        "stream": False,
        "format": "json",
    }
    options: dict[str, Any] = {
        "num_predict": profile.max_tokens,
        "num_ctx": ollama_num_ctx(
            system_prompt,
            _combined_user_prompt(cacheable_user_prefix, user_prompt),
            profile.max_tokens,
        ),
    }
    if not profile.omit_temperature:
        options["temperature"] = profile.temperature
    body["options"] = options
    r = requests.post(f"{base}/api/chat", json=body, timeout=timeout)
    if not r.ok:
        raise ProviderError(_http_error_text(r))
    text = (r.json().get("message", {}) or {}).get("content", "") or ""
    if not str(text).strip():
        raise ProviderError("Ollama returned an empty response.")
    return str(text)


def _http_error_text(r: requests.Response) -> str:
    try:
        data = r.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err)
        return str(err)
    except Exception:
        return r.text or f"HTTP {r.status_code}"


def _extract_openai_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("Model returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
            elif block.get("text"):
                parts.append(str(block["text"]))
        text = "".join(parts).strip()
    else:
        text = ""
    refusal = message.get("refusal")
    if refusal:
        raise ProviderError(f"Model refused the request: {refusal}")
    if not text:
        finish = choices[0].get("finish_reason") or ""
        hint = f" (finish_reason={finish})" if finish else ""
        raise ProviderError(f"Model returned an empty response{hint}.")
    return text
