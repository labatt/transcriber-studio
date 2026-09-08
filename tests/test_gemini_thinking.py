# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gemini must not spend the answer's budget on thinking, and must say when it did.

Measured against a real cleanup batch: thinking took 15,728 tokens of a 16,384
budget, leaving 656 for the answer. The JSON truncated mid-object, failed to
parse, and was read as "the batch is too big" — so the batch was halved and the
same 15,728 tokens were paid for again, twice. The batch size was never the
problem.

Two things follow. Cleanup is mechanical restructuring against a fixed schema,
so thinking is turned off for it; and a truncated answer has to be reported as
truncated, because "malformed JSON" is the one diagnosis that leads away from
the cause.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from transcriber_studio import ai_providers
from transcriber_studio.ai_store import ModelProfile


def _profile(max_tokens: int = 32_768) -> ModelProfile:
    return ModelProfile(
        provider="google", model_id="gemini-flash-latest",
        max_tokens=max_tokens, temperature=0.2,
    )


class _Response:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text

    def json(self):
        return self._payload


def _answer(text="{}", finish="STOP", usage=None):
    return {
        "candidates": [{
            "finishReason": finish,
            "content": {"parts": [{"text": text}]},
        }],
        "usageMetadata": usage or {},
    }


def _call(monkeypatch, responses, model="gemini-flash-latest", profile=None):
    """Run a Gemini chat, returning (sent bodies, result-or-exception)."""
    sent: list[dict] = []
    queue = list(responses)

    def fake_post(url, params=None, json=None, timeout=None):
        # A copy, not the object: the retry edits the body in place, so a
        # reference here would show the second request's shape for both.
        sent.append(copy.deepcopy(json))
        return queue.pop(0)

    monkeypatch.setattr(ai_providers.requests, "post", fake_post)
    settings = SimpleNamespace(ai_key_google="key")
    try:
        result = ai_providers.chat_completion(
            settings=settings, provider="google", model=model,
            system_prompt="sys", user_prompt="user",
            profile=profile or _profile(),
        )
    except Exception as e:      # noqa: BLE001 — the error is what is under test
        return sent, e
    return sent, result


# ---- turning thinking off --------------------------------------------
def test_thinking_is_turned_off_for_a_thinking_model(monkeypatch):
    sent, result = _call(monkeypatch, [_Response(_answer('{"ok":1}'))])
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert result == '{"ok":1}'


@pytest.mark.parametrize(
    "model", ["gemini-2.5-flash", "gemini-flash-latest", "gemini-pro-latest", "gemini-3-pro"],
)
def test_the_models_that_think_are_recognised(model):
    assert ai_providers.google_supports_thinking_config(model)


@pytest.mark.parametrize("model", ["gemini-1.5-pro", "gemini-1.0-pro"])
def test_older_models_are_not_sent_the_field(model, monkeypatch):
    sent, _result = _call(monkeypatch, [_Response(_answer())], model=model)
    assert "thinkingConfig" not in sent[0]["generationConfig"]


def test_a_model_that_rejects_the_field_is_asked_again_without_it(monkeypatch):
    """Refusing to answer because we asked for less work would be absurd."""
    sent, result = _call(monkeypatch, [
        _Response({}, status_code=400, text="Unknown name 'thinkingConfig'"),
        _Response(_answer('{"ok":1}')),
    ])
    assert len(sent) == 2
    assert "thinkingConfig" in sent[0]["generationConfig"]
    assert "thinkingConfig" not in sent[1]["generationConfig"]
    assert result == '{"ok":1}'


def test_an_unrelated_400_is_not_retried(monkeypatch):
    sent, error = _call(monkeypatch, [
        _Response({"error": {"message": "API key not valid"}},
                  status_code=400, text="API key not valid"),
    ])
    assert len(sent) == 1
    assert isinstance(error, ai_providers.ProviderError)


# ---- saying what actually happened -----------------------------------
def test_a_truncated_answer_is_reported_as_truncated(monkeypatch):
    """Not as malformed JSON, which sends the caller after the wrong cause."""
    _sent, error = _call(monkeypatch, [
        _Response(_answer('{"segments": [{"from_ind', finish="MAX_TOKENS")),
    ])
    assert isinstance(error, ai_providers.ProviderError)
    assert "ran out of output budget" in str(error)


def test_the_truncation_message_still_asks_for_a_smaller_batch(monkeypatch):
    """When the answer genuinely does not fit, splitting is the right move, and
    the cleanup retry logic decides that by matching on this marker."""
    from transcriber_studio.ai_cleanup import _should_split_chunk

    _sent, error = _call(monkeypatch, [
        _Response(_answer("{", finish="MAX_TOKENS")),
    ])
    assert _should_split_chunk(str(error))


def test_a_content_filter_is_named(monkeypatch):
    _sent, error = _call(monkeypatch, [_Response(_answer("", finish="SAFETY"))])
    assert "content filter" in str(error)
    assert "SAFETY" in str(error)


def test_an_unknown_stop_reason_is_still_reported(monkeypatch):
    _sent, error = _call(monkeypatch, [_Response(_answer("x", finish="OTHER"))])
    assert "stopped early" in str(error)
    assert "OTHER" in str(error)


def test_a_normal_answer_is_not_treated_as_a_problem(monkeypatch):
    _sent, result = _call(monkeypatch, [_Response(_answer('{"segments":[]}'))])
    assert result == '{"segments":[]}'


def test_a_missing_finish_reason_is_not_treated_as_a_problem(monkeypatch):
    """Not every response carries one; absence is not failure."""
    payload = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    _sent, result = _call(monkeypatch, [_Response(payload)])
    assert result == "{}"


# ---- the budgets agree -----------------------------------------------
def test_the_batch_budget_matches_what_is_actually_requested():
    """These used to disagree: batches were sized for 8,192 output tokens while
    the request asked for 16,384."""
    from transcriber_studio.ai_cleanup import output_token_ceiling

    assert output_token_ceiling("google", "gemini-flash-latest") == _profile().max_tokens
