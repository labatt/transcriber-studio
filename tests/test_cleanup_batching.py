# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""AI Cleanup batches must fit their answers, and a batch that does not must
be split at once rather than retried.

Measured live, 2026-09-08: a 503-segment, 95k-character batch was budgeted as
if its answer would be a quarter of that, was sent whole, came back cut off at
the output ceiling, was retried four more times with "adjusted settings" (35 to
45 seconds and a full answer's worth of tokens each), and only then split in
half, whereupon both halves succeeded. Three batches in a row did the same.
"""

from __future__ import annotations

import pytest

from transcriber_studio import ai_cleanup
from transcriber_studio.ai_store import ModelProfile, suggest_profile_fix
from transcriber_studio.models import Segment

TRUNCATED = ("finish_reason=max_tokens — gemini-flash-latest ran out of output budget "
             "before it finished the JSON, so the answer is cut off.")


def _segments(n: int, chars: int) -> list[Segment]:
    return [Segment(i, i + 1, "x" * chars, speaker="Speaker 1") for i in range(n)]


def _profile(max_tokens: int = 32_768) -> ModelProfile:
    return ModelProfile(provider="google", model_id="gemini-flash-latest",
                        max_tokens=max_tokens)


# ---- sizing --------------------------------------------------------------------
def test_the_answer_is_budgeted_at_the_full_text_not_a_quarter_of_it():
    seg = Segment(0, 1, "a" * 400)
    assert ai_cleanup._estimate_segment_output_chars(seg) >= 400


def test_a_95k_transcript_no_longer_goes_up_as_one_batch_under_a_45k_answer_budget():
    """The failing case, with the old 16,384-token budget: must split."""
    budget = ai_cleanup.ChunkBudget(
        max_output_chars=int(16_384 * ai_cleanup.CHARS_PER_TOKEN * ai_cleanup.OUTPUT_SAFETY),
        max_input_chars=120_000, max_segments=600, output_token_ceiling=16_384,
    )
    chunks = ai_cleanup._chunk_segments(_segments(503, 150), budget)   # ~95k of text
    assert len(chunks) >= 2
    for chunk in chunks:
        assert sum(ai_cleanup._estimate_segment_output_chars(s) for s in chunk) <= budget.max_output_chars


def test_the_gemini_budget_takes_that_transcript_in_two_or_three_batches():
    budget = ai_cleanup.chunk_budget("google", "gemini-flash-latest")
    assert budget.output_token_ceiling == 32_768
    chunks = ai_cleanup._chunk_segments(_segments(503, 150), budget)
    assert 1 <= len(chunks) <= 3


# ---- what a truncated answer means ---------------------------------------------
def test_gemini_truncation_is_recognised_as_truncation():
    assert ai_cleanup._is_truncation(TRUNCATED)
    assert ai_cleanup._is_truncation("stop_reason=max_tokens")
    assert not ai_cleanup._is_truncation("temperature is not supported")


def test_a_truncated_batch_is_split_after_one_send_not_five(monkeypatch):
    sends = []

    def cut_off(*args, **kwargs):
        sends.append(1)
        raise RuntimeError(TRUNCATED)

    monkeypatch.setattr(ai_cleanup, "_chat_completion_cancellable", cut_off)
    monkeypatch.setattr(ai_cleanup, "save_profile", lambda *_a, **_k: None)
    lines = []
    with pytest.raises(RuntimeError) as excinfo:
        ai_cleanup._cleanup_chunk_once(
            _segments(4, 10), ["Speaker 1"], object(), "google", "gemini-flash-latest",
            _profile(), "", lines.append,
        )
    assert len(sends) == 1, "no point retrying a batch whose answer cannot fit"
    assert ai_cleanup._should_split_chunk(str(excinfo.value)), "the caller splits on this"
    assert any("cut off at the output limit" in line for line in lines)


def test_a_single_segment_that_is_cut_off_is_still_reported_not_split(monkeypatch):
    monkeypatch.setattr(
        ai_cleanup, "_chat_completion_cancellable",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(TRUNCATED)),
    )
    monkeypatch.setattr(ai_cleanup, "save_profile", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError):
        ai_cleanup._cleanup_chunk_once(
            _segments(1, 10), ["Speaker 1"], object(), "google", "gemini-flash-latest",
            _profile(), "", None,
        )


def test_the_profile_fixer_does_not_halve_max_tokens_for_a_truncated_answer():
    """That branch is for a model rejecting max_tokens as too large. A cut-off
    answer needs more room or a smaller batch, never less room."""
    at_cap = _profile(16_384)
    assert suggest_profile_fix(TRUNCATED, at_cap, provider="google") is None


def test_the_profile_fixer_still_raises_a_small_budget_first():
    small = _profile(4_096)
    fixed = suggest_profile_fix(TRUNCATED, small, provider="google")
    assert fixed is not None and fixed.max_tokens > small.max_tokens


def test_the_split_log_line_says_why(monkeypatch):
    """"batch failed — splitting" told the user nothing about the cause."""
    calls = []

    def once(segments, *args, **kwargs):
        calls.append(len(segments))
        if len(segments) > 1:
            raise RuntimeError(TRUNCATED)
        return list(segments)

    monkeypatch.setattr(ai_cleanup, "_cleanup_chunk_once", once)
    lines = []
    ai_cleanup._cleanup_chunk(
        _segments(2, 10), ["Speaker 1"], object(), "google", "gemini-flash-latest",
        _profile(), "", lines.append,
    )
    assert calls == [2, 1, 1]
    assert any("finish_reason=max_tokens" in line and "splitting 2" in line for line in lines)


# ---- a fix learned on batch 1 holds for batch 2 --------------------------------
def test_a_parameter_fix_learned_on_one_batch_is_not_relearned_on_the_next(monkeypatch):
    """Live: every one of four batches to gpt-5.6-terra hit "switching to
    max_completion_tokens" and "model rejected temperature" again, because each
    batch started from the profile loaded before the run."""
    sends = []

    def picky(settings, provider, model, system, user, profile, **kwargs):
        sends.append(profile)
        if not profile.omit_temperature:
            raise RuntimeError("Unsupported value: 'temperature' does not support 0.2 with this model.")
        return '{"segments": []}'

    monkeypatch.setattr(ai_cleanup, "_chat_completion_cancellable", picky)
    monkeypatch.setattr(ai_cleanup, "save_profile", lambda *_a, **_k: None)
    monkeypatch.setattr(ai_cleanup, "_LEARNED_PROFILES", {})
    monkeypatch.setattr(ai_cleanup, "_apply_cleanup", lambda segments, parsed: list(segments))
    stale = _profile()

    ai_cleanup._cleanup_chunk_once(_segments(2, 5), ["Speaker 1"], object(), "openai",
                                   "gpt-5.6-terra", stale, "", None)
    ai_cleanup._cleanup_chunk_once(_segments(2, 5), ["Speaker 1"], object(), "openai",
                                   "gpt-5.6-terra", stale, "", None)

    assert len(sends) == 3, "two sends for the first batch, one for the second"
    assert sends[2].omit_temperature


def test_gpt_5_models_get_the_larger_budget_and_gpt_4_does_not():
    """OpenAI documents 128,000 max output tokens for gpt-5.6-terra; the app
    asks for a quarter of that, matching Gemini, for the timeout's sake."""
    assert ai_cleanup.output_token_ceiling("openai", "gpt-5.6-terra") == 32_768
    assert ai_cleanup.output_token_ceiling("openrouter", "openai/gpt-5.6-terra") == 32_768
    assert ai_cleanup.output_token_ceiling("openai", "gpt-4.1") == 16_384


# ---- OpenAI's reasoning models, built in rather than learned by rejection ------
def test_a_gpt_5_6_model_starts_with_the_parameters_it_accepts():
    from transcriber_studio.ai_store import default_profile

    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "gpt-6-astra"):
        p = default_profile("openai", model)
        assert p.use_max_completion_tokens and p.omit_temperature, model
    via_router = default_profile("openrouter", "openai/gpt-5.6-luna")
    assert via_router.use_max_completion_tokens and via_router.omit_temperature


def test_older_and_other_vendors_models_keep_the_plain_defaults():
    from transcriber_studio.ai_store import default_profile

    for provider, model in (("openai", "gpt-4.1"), ("openrouter", "anthropic/claude-opus-5"),
                            ("google", "gemini-flash-latest")):
        p = default_profile(provider, model)
        assert not p.use_max_completion_tokens and not p.omit_temperature, model


def test_the_1m_context_family_gets_the_larger_input_ceiling():
    assert ai_cleanup.input_char_ceiling("openai", "gpt-5.6-luna") == 900_000
    assert ai_cleanup.input_char_ceiling("openai", "gpt-6-astra") == 900_000
    assert ai_cleanup.input_char_ceiling("openai", "gpt-4.1") == 120_000
    assert ai_cleanup.output_token_ceiling("openai", "gpt-6-astra") == 32_768


def test_reasoning_effort_is_low_for_a_gpt_5_6_model_and_absent_for_gpt_4(monkeypatch):
    from types import SimpleNamespace

    from transcriber_studio import ai_providers
    from transcriber_studio.ai_store import default_profile

    sent = []

    class _R:
        status_code, ok, text = 200, True, ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"segments": []}'}}], "usage": {}}

    monkeypatch.setattr(ai_providers.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: sent.append(json) or _R())
    monkeypatch.setattr(ai_providers, "_openai_compat_headers", lambda *a, **k: {})
    settings = SimpleNamespace(ai_key_openai="k")
    fn = ai_providers._chat_openai_compat

    fn(settings=settings, provider="openai", model="gpt-5.6-terra", system_prompt="sys", user_prompt="user",
       profile=default_profile("openai", "gpt-5.6-terra"))
    fn(settings=settings, provider="openai", model="gpt-4.1", system_prompt="sys", user_prompt="user",
       profile=default_profile("openai", "gpt-4.1"))

    terra, four = sent
    assert terra["reasoning_effort"] == ai_providers.CLEANUP_REASONING_EFFORT
    assert "max_completion_tokens" in terra and "temperature" not in terra
    assert "reasoning_effort" not in four and "max_tokens" in four and "temperature" in four


def test_a_model_that_rejects_reasoning_effort_is_asked_again_without_it(monkeypatch):
    from types import SimpleNamespace

    from transcriber_studio import ai_providers
    from transcriber_studio.ai_store import default_profile

    sent = []

    class _Bad:
        status_code, ok = 400, False
        text = '{"error": {"message": "Unsupported parameter: reasoning_effort"}}'

        @staticmethod
        def json():
            return {"error": {"message": "Unsupported parameter: reasoning_effort"}}

    class _Good:
        status_code, ok, text = 200, True, ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"segments": []}'}}], "usage": {}}

    answers = [_Bad(), _Good()]
    monkeypatch.setattr(ai_providers.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: sent.append(dict(json)) or answers.pop(0))
    monkeypatch.setattr(ai_providers, "_openai_compat_headers", lambda *a, **k: {})
    out = ai_providers._chat_openai_compat(
        settings=SimpleNamespace(ai_key_openai="k"), provider="openai", model="gpt-5.6-luna",
        system_prompt="sys", user_prompt="user", profile=default_profile("openai", "gpt-5.6-luna"),
    )
    assert out == '{"segments": []}'
    assert "reasoning_effort" in sent[0] and "reasoning_effort" not in sent[1]
