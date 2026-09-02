# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ollama must be told how much context to allocate.

It defaults to roughly 4k tokens and drops the overflow without complaining, so
a cleanup batch would arrive as a fragment. The model then answers for the
fragment, the answer comes back short, and the caller reads that as a batch too
big to handle — halving it forever, because the batch size was never the cause.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from transcriber_studio import ai_providers
from transcriber_studio.ai_cleanup import input_char_ceiling, output_token_ceiling

OLLAMA_PROVIDERS = ("ollama_local", "ollama_cloud")


def test_a_short_prompt_does_not_allocate_a_huge_window():
    """Context costs KV-cache memory on the user's own machine."""
    assert ai_providers.ollama_num_ctx("system", "hello", 512) == \
        ai_providers.OLLAMA_MIN_NUM_CTX


def test_the_window_grows_with_the_prompt():
    small = ai_providers.ollama_num_ctx("s", "u" * 1_000, 512)
    large = ai_providers.ollama_num_ctx("s", "u" * 200_000, 512)
    assert large > small
    assert large <= ai_providers.OLLAMA_MAX_NUM_CTX


def test_the_window_leaves_room_for_the_answer():
    """num_predict tokens have to fit alongside the prompt, not instead of it."""
    prompt = "u" * 20_000
    lean = ai_providers.ollama_num_ctx("s", prompt, 256)
    verbose = ai_providers.ollama_num_ctx("s", prompt, 16_384)
    assert verbose > lean


@pytest.mark.parametrize("provider", OLLAMA_PROVIDERS)
def test_a_full_batch_fits_the_window_it_asks_for(provider):
    """The batch ceiling and the context we request must agree.

    If the ceiling lets through more than num_ctx can hold, the overflow is
    dropped in silence — which is the bug this whole module exists to prevent.
    """
    model = "qwen3:8b"
    chars = input_char_ceiling(provider, model)
    output_tokens = output_token_ceiling(provider, model)
    system = "s" * 2_000       # the cleanup system prompt is not small
    window = ai_providers.ollama_num_ctx(system, "u" * chars, output_tokens)

    needed = (
        (len(system) + chars) / ai_providers.OLLAMA_CHARS_PER_TOKEN
        + output_tokens
        + ai_providers.OLLAMA_CTX_HEADROOM_TOKENS
    )
    assert needed <= window, (
        f"{provider}: a full batch needs ~{needed:.0f} tokens "
        f"but only {window} were requested"
    )


@pytest.mark.parametrize("provider", OLLAMA_PROVIDERS)
def test_the_ceiling_is_not_the_unrelated_fallback(provider):
    """Ollama used to fall through to the 60k default meant for unknown providers."""
    assert input_char_ceiling(provider, "qwen3:8b") != 60_000


def _captured_body(monkeypatch, chat_fn, provider_key):
    seen = {}

    class _Response:
        ok = True

        @staticmethod
        def json():
            return {"message": {"content": '{"segments": []}'}}

    def fake_post(url, json=None, **kwargs):
        seen["body"] = json
        return _Response()

    monkeypatch.setattr(ai_providers.requests, "post", fake_post)
    settings = SimpleNamespace(
        ollama_local_url="http://localhost:11434",
        ai_key_ollama_cloud="key",
    )
    profile = ai_providers.ModelProfile(
        provider=provider_key, model_id="qwen3:8b",
        temperature=0.2, max_tokens=4_096, omit_temperature=False,
    )
    chat_fn(
        settings=settings,
        provider=provider_key,
        model="qwen3:8b",
        system_prompt="s" * 2_000,
        user_prompt="u" * 50_000,
        profile=profile,
    )
    return seen["body"]


def test_local_requests_carry_num_ctx(monkeypatch):
    body = _captured_body(
        monkeypatch, ai_providers._chat_ollama_local, "ollama_local"
    )
    assert body["options"]["num_ctx"] > ai_providers.OLLAMA_MIN_NUM_CTX
    assert body["options"]["num_predict"] == 4_096
    assert body["options"]["temperature"] == 0.2


def test_cloud_requests_carry_num_ctx(monkeypatch):
    body = _captured_body(
        monkeypatch, ai_providers._chat_ollama_cloud, "ollama_cloud"
    )
    assert body["options"]["num_ctx"] > ai_providers.OLLAMA_MIN_NUM_CTX


def test_omitting_temperature_still_sends_num_ctx(monkeypatch):
    """Some models reject temperature; the context window is not optional."""
    seen = {}

    class _Response:
        ok = True

        @staticmethod
        def json():
            return {"message": {"content": "{}"}}

    monkeypatch.setattr(
        ai_providers.requests, "post",
        lambda url, json=None, **kw: (seen.update(body=json), _Response())[1],
    )
    ai_providers._chat_ollama_local(
        settings=SimpleNamespace(ollama_local_url="http://localhost:11434"),
        provider="ollama_local",
        model="qwen3:8b",
        system_prompt="s",
        user_prompt="u" * 40_000,
        profile=ai_providers.ModelProfile(
            provider="ollama_local", model_id="qwen3:8b",
            temperature=0.2, max_tokens=2_048, omit_temperature=True,
        ),
    )
    assert "temperature" not in seen["body"]["options"]
    assert seen["body"]["options"]["num_ctx"] > ai_providers.OLLAMA_MIN_NUM_CTX
