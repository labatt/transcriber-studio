# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The three layers in front of the decoder: denoise, VAD, vocabulary biasing."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from transcriber_studio import components, denoise, vad, vocab_bias, whisper_models

components.refresh_path()   # a winget upgrade moves ffmpeg; find it where it is now
from tests.support import isolated_denoise_cache, isolated_glossary_dir
from transcriber_studio.audio_utils import FFMPEG, have_ffmpeg, probe
from transcriber_studio.config import Settings
from transcriber_studio.transcriber import TranscribeOptions, transcribe_kwargs

# ---- picking a denoiser ------------------------------------------------


def _settings(**over) -> Settings:
    s = Settings(denoise_enabled=True)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_denoise_off_resolves_to_nothing():
    assert denoise.resolve(Settings(denoise_enabled=False)) == denoise.NONE


def test_auto_prefers_the_binary_then_the_package_then_ffmpeg(monkeypatch):
    s = _settings(denoise_backend=denoise.AUTO)
    available = {denoise.DEEP_FILTER, denoise.PYTHON_DF, denoise.FFMPEG_DN}
    monkeypatch.setattr(denoise, "is_available", lambda b, _s: b in available)

    assert denoise.resolve(s) == denoise.DEEP_FILTER
    available.discard(denoise.DEEP_FILTER)
    assert denoise.resolve(s) == denoise.PYTHON_DF
    available.discard(denoise.PYTHON_DF)
    assert denoise.resolve(s) == denoise.FFMPEG_DN


def test_an_explicitly_chosen_backend_is_never_silently_swapped(monkeypatch):
    """Asking for DeepFilterNet and quietly getting afftdn would be a lie."""
    s = _settings(denoise_backend=denoise.DEEP_FILTER)
    monkeypatch.setattr(denoise, "is_available", lambda b, _s: b == denoise.FFMPEG_DN)

    assert denoise.resolve(s) == denoise.NONE
    assert "not on PATH" in denoise.describe(s)


def test_describe_says_when_the_fallback_is_standing_in(monkeypatch):
    s = _settings(denoise_backend=denoise.AUTO)
    monkeypatch.setattr(denoise, "is_available", lambda b, _s: b == denoise.FFMPEG_DN)

    text = denoise.describe(s)
    assert "ffmpeg" in text and "deep-filter" in text


def test_deep_filter_command_always_compensates_the_delay():
    """Without -D the enhanced audio is shifted and every timestamp moves."""
    cmd = denoise.deep_filter_command(
        "deep-filter.exe", Path("in.wav"), Path("out"), model_path="", postfilter=False
    )

    assert cmd[0] == "deep-filter.exe"
    assert "-D" in cmd
    assert cmd[cmd.index("-o") + 1] == "out"
    assert cmd[-1] == "in.wav"
    assert "--pf" not in cmd


def test_deep_filter_command_passes_a_model_and_postfilter_when_asked():
    cmd = denoise.deep_filter_command(
        "deep-filter", Path("in.wav"), Path("out"), model_path=" DFN3.tar.gz ",
        postfilter=True,
    )

    assert cmd[cmd.index("-m") + 1] == "DFN3.tar.gz"
    assert "--pf" in cmd


def test_full_suppression_is_left_to_the_tools_own_default():
    """Passing its default back to it is noise on the command line."""
    cmd = denoise.deep_filter_command(
        "df", Path("in.wav"), Path("out"), atten_lim_db=100
    )
    assert "-a" not in cmd


def test_a_reduced_noise_limit_is_passed_through():
    cmd = denoise.deep_filter_command(
        "df", Path("in.wav"), Path("out"), atten_lim_db=40
    )
    assert cmd[cmd.index("-a") + 1] == "40"


def test_cache_key_follows_the_settings_that_change_the_output(tmp_path):
    source = tmp_path / "a.wav"
    source.write_bytes(b"x" * 32)
    base = _settings()
    other = _settings(denoise_postfilter=True)
    quieter = _settings(denoise_atten_lim_db=40)

    same = denoise.cache_key(str(source), denoise.FFMPEG_DN, base)
    assert same == denoise.cache_key(str(source), denoise.FFMPEG_DN, base)
    assert same != denoise.cache_key(str(source), denoise.DEEP_FILTER, base)
    assert same != denoise.cache_key(str(source), denoise.FFMPEG_DN, other)
    assert same != denoise.cache_key(str(source), denoise.FFMPEG_DN, quieter)


def test_a_failing_backend_leaves_the_original_audio_in_place(monkeypatch, tmp_path):
    source = tmp_path / "a.wav"
    source.write_bytes(b"x" * 32)
    with isolated_denoise_cache():
        monkeypatch.setattr(denoise, "resolve", lambda _s: denoise.FFMPEG_DN)
        monkeypatch.setattr(
            denoise, "_ffmpeg_denoise",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no codec")),
        )
        logs: list[str] = []

        out = denoise.enhance(str(source), _settings(), log_cb=logs.append)

        assert out == str(source)
        assert any("no codec" in line for line in logs)


@pytest.mark.skipif(not have_ffmpeg(), reason="needs ffmpeg")
def test_ffmpeg_backend_produces_mono_16k_audio_and_reuses_it():
    with isolated_denoise_cache(), tempfile.TemporaryDirectory() as tmp:
        noisy = Path(tmp) / "noisy.wav"
        subprocess.run(
            [
                FFMPEG, "-y",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                "-f", "lavfi", "-i", "anoisesrc=d=3:c=pink:a=0.4",
                "-filter_complex", "[0][1]amix=inputs=2:duration=shortest",
                "-ac", "2", "-ar", "44100", str(noisy),
            ],
            capture_output=True, check=True,
        )
        s = _settings(denoise_backend=denoise.FFMPEG_DN)

        first = denoise.enhance(str(noisy), s)
        meta = probe(first)
        assert first != str(noisy)
        assert meta["channels"] == 1
        assert round(meta["duration"]) == 3      # denoising must not shift the timeline

        logs: list[str] = []
        second = denoise.enhance(str(noisy), s, log_cb=logs.append)
        assert second == first
        assert any("reusing" in line for line in logs)


def test_prune_keeps_only_the_newest_enhanced_files():
    with isolated_denoise_cache() as cache:
        cache.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            path = cache / f"{i}.wav"
            path.write_bytes(b"x")
            # Distinct mtimes, oldest first.
            import os

            os.utime(path, (1_000_000 + i, 1_000_000 + i))

        denoise.prune(keep=2)

        assert sorted(p.name for p in cache.glob("*.wav")) == ["3.wav", "4.wav"]


# ---- VAD ---------------------------------------------------------------


def test_vad_parameters_are_filtered_to_what_the_installed_version_takes(monkeypatch):
    """faster-whisper's VadOptions has gained and lost fields across releases."""
    monkeypatch.setattr(vad, "supported_fields", lambda: {"threshold", "speech_pad_ms"})

    params = vad.parameters(Settings(vad_threshold=0.4, vad_speech_pad_ms=250))

    assert params == {"threshold": 0.4, "speech_pad_ms": 250}


def test_an_unset_speech_cap_is_left_out_entirely():
    params = vad.parameters(Settings(vad_max_speech_s=0.0))
    assert "max_speech_duration_s" not in params

    params = vad.parameters(Settings(vad_max_speech_s=30.0))
    assert params.get("max_speech_duration_s") == 30.0


def test_vad_describe_names_the_version_that_will_run():
    text = vad.describe(Settings(vad_enabled=True))
    assert "Silero VAD" in text
    assert "threshold" in text

    assert "off" in vad.describe(Settings(vad_enabled=False))


# ---- vocabulary biasing ------------------------------------------------


def test_terms_come_from_the_shared_glossary_the_job_uses():
    from transcriber_studio import glossary_store

    with isolated_glossary_dir():
        shared = glossary_store.create(
            "Acme",
            terms=[
                {"canonical": "GrowthMark", "variants": [], "type": "product"},
                {"canonical": "Dana Reyes", "variants": [], "type": "person"},
            ],
        )
        s = Settings(glossary_shared_id=shared.id)

        terms = vocab_bias.collect_terms(s)

        # People first: a mis-heard name costs more than a mis-heard noun.
        assert terms == ["Dana Reyes", "GrowthMark"]


def test_typed_terms_outrank_extracted_ones_and_duplicates_collapse():
    with isolated_glossary_dir():
        s = Settings(bias_extra_terms="NorthGate\nGrowthMark")
        payload = {
            "speakers": [{"name": "Dana Reyes"}],
            "terms": [{"canonical": "growthmark", "type": "product"}],
        }

        terms = vocab_bias.collect_terms(s, extra_payloads=[payload])

        assert terms[:2] == ["NorthGate", "GrowthMark"]
        assert [t.lower() for t in terms].count("growthmark") == 1


def test_generic_labels_and_common_words_never_reach_the_prompt():
    with isolated_glossary_dir():
        s = Settings(bias_extra_terms="SPEAKER_00, the, meeting, X, Nyra")

        assert vocab_bias.collect_terms(s) == ["Nyra"]


def test_the_budget_drops_terms_from_the_tail_not_the_head():
    prompt = vocab_bias.build(["Alpha", "Bravo", "Charlie"], max_chars=14)

    assert prompt == "Alpha, Bravo"


def test_the_budget_is_capped_below_what_the_decoder_would_truncate():
    prompt = vocab_bias.build(["Word"] * 500, max_chars=100_000)

    assert len(prompt) <= vocab_bias.HARD_CHAR_CEILING


def test_biasing_off_produces_no_prompt():
    with isolated_glossary_dir():
        s = Settings(bias_enabled=False, bias_extra_terms="NorthGate")
        assert vocab_bias.hotwords(s) == ""


def test_summarize_reports_what_did_not_fit():
    terms = ["Alpha", "Bravo", "Charlie"]
    prompt = vocab_bias.build(terms, max_chars=14)

    assert "1 did not fit" in vocab_bias.summarize(terms, prompt)


# ---- what reaches faster-whisper ---------------------------------------


def test_the_three_layers_reach_the_decoder_call():
    opts = TranscribeOptions(
        vad_enabled=True,
        vad_parameters={"threshold": 0.4},
        hotwords="NorthGate, GrowthMark",
        hallucination_guard=True,
    )

    kwargs = transcribe_kwargs(opts, "en")

    assert kwargs["vad_filter"] is True
    assert kwargs["vad_parameters"] == {"threshold": 0.4}
    assert kwargs["hotwords"] == "NorthGate, GrowthMark"
    # Carried-over context is what turns one hallucination into a paragraph.
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["hallucination_silence_threshold"] == opts.hallucination_silence_s


def test_the_guard_is_the_only_thing_that_turns_off_carry_over():
    kwargs = transcribe_kwargs(TranscribeOptions(hallucination_guard=False), None)

    assert "condition_on_previous_text" not in kwargs
    assert "hallucination_silence_threshold" not in kwargs


def test_the_silence_threshold_needs_word_timestamps_to_mean_anything():
    kwargs = transcribe_kwargs(
        TranscribeOptions(hallucination_guard=True, word_timestamps=False), None
    )

    assert kwargs["condition_on_previous_text"] is False
    assert "hallucination_silence_threshold" not in kwargs


def test_no_vocabulary_means_no_hotwords_argument():
    assert "hotwords" not in transcribe_kwargs(TranscribeOptions(hotwords=""), None)


# ---- the model list ----------------------------------------------------


def test_crisperwhisper_is_offered_and_loads_as_a_ctranslate2_repo():
    assert whisper_models.CRISPER in whisper_models.ORDER
    assert whisper_models.CRISPER == "nyralabs/faster_CrisperWhisper"
    assert whisper_models.label(whisper_models.CRISPER).startswith("CrisperWhisper")


def test_crisperwhispers_description_states_its_limits():
    text = whisper_models.describe(whisper_models.CRISPER, has_gpu=True)

    assert "verbatim" in text.lower()
    assert "English and German" in text          # the fine-tune's language coverage
    assert "timestamps" in text.lower()          # the CTranslate2 caveat


def test_the_stock_sizes_still_describe_themselves():
    assert whisper_models.label("large-v3") == "large-v3"
    assert "Recommended" in whisper_models.describe("large-v3", has_gpu=True)
