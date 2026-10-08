# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""One run's choices, applied to a copy and never saved."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from transcriber_studio.config import Settings
from transcriber_studio.run_plan import RunPlan
from transcriber_studio.ui.run_options_dialog import RunOptionsDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_apply_leaves_the_original_settings_alone():
    s = Settings(denoise_enabled=True, ai_cleanup_enabled=True, min_speakers=2, max_speakers=6)
    run = RunPlan(denoise=False, ai_cleanup=False, speakers=3).apply(s)
    assert (run.denoise_enabled, run.ai_cleanup_enabled) == (False, False)
    assert (run.min_speakers, run.max_speakers) == (3, 3)
    assert (s.denoise_enabled, s.ai_cleanup_enabled) == (True, True)
    assert (s.min_speakers, s.max_speakers) == (2, 6)


def test_auto_speaker_count_keeps_the_panels_bounds():
    s = Settings(min_speakers=2, max_speakers=6)
    run = RunPlan(speakers=0).apply(s)
    assert (run.min_speakers, run.max_speakers) == (2, 6)


def test_identifying_speakers_implies_detecting_them():
    run = RunPlan(identify_speakers=True, detect_speakers=False).apply(Settings())
    assert run.diarization_enabled is True


def test_from_settings_reads_an_exact_count_only():
    assert RunPlan.from_settings(Settings(min_speakers=4, max_speakers=4)).speakers == 4
    assert RunPlan.from_settings(Settings(min_speakers=2, max_speakers=6)).speakers == 0
    assert RunPlan.from_settings(Settings(min_speakers=0, max_speakers=3)).speakers == 0


def test_describe_mentions_each_choice():
    text = RunPlan(identify_speakers=True, speakers=3, denoise=True).describe()
    assert "identify speakers first (3 people)" in text
    assert "denoise" in text and "no AI cleanup" in text


# ---- the dialog ------------------------------------------------------
def test_identify_is_offered_only_when_pyannote_can_run(app, monkeypatch):
    from transcriber_studio import diarization

    monkeypatch.setattr(diarization, "is_available", lambda: True)
    dlg = RunOptionsDialog(Settings(hf_token=""), [])
    assert not dlg.identify.isEnabled()
    dlg = RunOptionsDialog(Settings(hf_token="hf_x"), [])
    assert dlg.identify.isEnabled()


def test_identify_is_not_offered_on_gemini(app, monkeypatch):
    from transcriber_studio import diarization

    monkeypatch.setattr(diarization, "is_available", lambda: True)
    dlg = RunOptionsDialog(Settings(hf_token="hf_x", stt_engine="gemini"), [])
    assert not dlg.identify.isEnabled()
    assert dlg.plan().identify_speakers is False


def test_ticking_identify_locks_detection_on(app, monkeypatch):
    from transcriber_studio import diarization

    monkeypatch.setattr(diarization, "is_available", lambda: True)
    dlg = RunOptionsDialog(Settings(hf_token="hf_x", diarization_enabled=False), [])
    dlg.identify.setChecked(True)
    assert dlg.detect.isChecked() and not dlg.detect.isEnabled()
    dlg.speakers.setValue(5)
    plan = dlg.plan()
    assert plan.identify_speakers and plan.detect_speakers and plan.speakers == 5
    assert plan.bounds == (5, 5)


def test_identify_only_stops_short_of_transcribing():
    plan = RunPlan(identify_speakers=True, identify_only=True, ai_cleanup=True, speakers=2)
    run = plan.apply(Settings(ai_cleanup_provider="openai", ai_cleanup_model="gpt"))
    assert run.diarization_enabled is True
    assert run.ai_cleanup_enabled is False, "nothing to clean up when nothing is transcribed"
    assert "no transcription" in plan.describe()


def test_identify_only_is_offered_under_identify(app, monkeypatch):
    from transcriber_studio import diarization

    monkeypatch.setattr(diarization, "is_available", lambda: True)
    dlg = RunOptionsDialog(
        Settings(hf_token="hf_x", ai_cleanup_provider="openai", ai_cleanup_model="gpt"), [],
    )
    assert not dlg.identify_only.isEnabled()
    dlg.identify.setChecked(True)
    assert dlg.identify_only.isEnabled() and dlg.cleanup.isEnabled()
    dlg.identify_only.setChecked(True)
    assert not dlg.cleanup.isEnabled()
    plan = dlg.plan()
    assert plan.identify_speakers and plan.identify_only
    # Unticking identify takes the stop-there option with it.
    dlg.identify.setChecked(False)
    assert not dlg.identify_only.isChecked() and not dlg.identify_only.isEnabled()
    assert dlg.plan().identify_only is False


def test_a_minute_limit_rides_on_the_run_copy_only():
    s = Settings()
    run = RunPlan(limit_minutes=10).apply(s)
    assert run.limit_minutes == 10 and s.limit_minutes == 0
    assert RunPlan(limit_minutes=-3).apply(s).limit_minutes == 0
    assert "first 10 minutes only" in RunPlan(limit_minutes=10).describe()


def test_the_limit_box_defaults_to_ten_minutes_and_is_off(app):
    dlg = RunOptionsDialog(Settings(), [])
    assert not dlg.limit.isChecked() and not dlg.limit_minutes.isEnabled()
    assert dlg.plan().limit_minutes == 0
    dlg.limit.setChecked(True)
    assert dlg.limit_minutes.isEnabled() and dlg.limit_minutes.value() == 10
    dlg.limit_minutes.setValue(3)
    assert dlg.plan().limit_minutes == 3


def test_cleanup_cannot_be_chosen_without_a_model(app):
    dlg = RunOptionsDialog(Settings(ai_cleanup_enabled=True, ai_cleanup_model=""), [])
    assert not dlg.cleanup.isEnabled()
    assert dlg.plan().ai_cleanup is False


# ---- naming the transcript and keeping the audio ---------------------
def test_a_transcript_name_and_audio_folder_ride_on_the_run_copy_only():
    s = Settings()
    plan = RunPlan(transcript_name="  Board call  ", save_audio=True, audio_dir="D:/keep")
    run = plan.apply(s)
    assert run.filename_override == "Board call" and run.save_audio_dir == "D:/keep"
    assert s.filename_override == "" and s.save_audio_dir == ""
    # Unticked, the folder in the box means nothing.
    assert RunPlan(save_audio=False, audio_dir="D:/keep").apply(s).save_audio_dir == ""
    text = plan.describe()
    assert "save the audio to D:/keep" in text and "transcript named 'Board call'" in text


def test_the_dialog_names_the_run_and_keeps_the_audio(app):
    from transcriber_studio.models import Recording, Source

    plaud = Recording(source=Source.PLAUD, id="p1", name="Weekly sync", date="2026-10-07")
    dlg = RunOptionsDialog(Settings(audio_download_dir="D:/Downloads"), [plaud])
    assert dlg.filename.text() == "" and dlg.filename.placeholderText() == "2026-10-07_Weekly sync"
    assert dlg.save_audio.isEnabled() and not dlg.save_audio.isChecked()
    assert dlg.audio_dir.text() == "D:/Downloads" and not dlg.audio_dir.isEnabled()
    plan = dlg.plan()
    assert plan.transcript_name == "" and plan.save_audio is False
    dlg.filename.setText("Board call.txt")
    dlg.save_audio.setChecked(True)
    assert dlg.audio_dir.isEnabled() and dlg.browse_btn.isEnabled()
    plan = dlg.plan()
    assert plan.transcript_name == "Board call.txt"
    assert plan.save_audio and plan.audio_dir == "D:/Downloads"


def test_saving_the_audio_is_offered_only_for_plaud_recordings(app):
    from transcriber_studio.models import Recording, Source

    local = Recording(source=Source.LOCAL, id="a.wav", name="a", local_path="a.wav")
    dlg = RunOptionsDialog(Settings(), [local])
    assert not dlg.save_audio.isEnabled()
    dlg.save_audio.setChecked(True)
    assert dlg.plan().save_audio is False


def test_identify_only_greys_out_naming_and_keeping(app, monkeypatch):
    from transcriber_studio import diarization
    from transcriber_studio.models import Recording, Source

    monkeypatch.setattr(diarization, "is_available", lambda: True)
    plaud = Recording(source=Source.PLAUD, id="p1", name="n", date="2026-10-07")
    dlg = RunOptionsDialog(Settings(hf_token="hf_x"), [plaud])
    dlg.filename.setText("x")
    dlg.save_audio.setChecked(True)
    dlg.identify.setChecked(True)
    dlg.identify_only.setChecked(True)
    assert not dlg.filename.isEnabled() and not dlg.save_audio.isEnabled()
    plan = dlg.plan()
    assert plan.transcript_name == "" and plan.save_audio is False
    dlg.identify_only.setChecked(False)
    assert dlg.filename.isEnabled() and dlg.save_audio.isEnabled()
    assert dlg.plan().transcript_name == "x" and dlg.plan().save_audio
