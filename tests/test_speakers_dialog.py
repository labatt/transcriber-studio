# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Managing enrolled voices, and the record that lets the thresholds be tuned.

The thresholds started as reasoning rather than measurement, so the match log
is the load-bearing part of this dialog: it is the only thing that can tell the
user whether the bar is in the right place for their people and their rooms.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from transcriber_studio import voiceprints
from transcriber_studio.config import Settings
from transcriber_studio.ui.speakers_dialog import SpeakersDialog

from .support import isolated_voiceprints

DIM = 8


def _vector(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIM - len(values))
    return [float(x) for x in padded[:DIM]]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ---- isolation -------------------------------------------------------
def test_the_match_log_follows_the_profile_directory():
    """Regression: it used to be built at import time from PROFILE_DIR, so a
    test that redirected PROFILE_DIR still wrote into the user's real history."""
    real = voiceprints.match_log_path()
    with isolated_voiceprints() as directory:
        assert voiceprints.match_log_path().parent == directory
        assert voiceprints.match_log_path() != real
    assert voiceprints.match_log_path() == real


def test_recording_matches_writes_only_inside_the_isolated_directory():
    with isolated_voiceprints() as directory:
        voiceprints.record_matches(
            [voiceprints.Match(label="SPEAKER_00", name="Alice", score=0.9)],
            source="a meeting",
        )
        assert (directory / voiceprints.MATCH_LOG_NAME).exists()


# ---- the log ---------------------------------------------------------
def test_a_match_round_trips_through_the_log():
    with isolated_voiceprints():
        voiceprints.record_matches([
            voiceprints.Match(label="SPEAKER_00", name="Alice", score=0.81,
                              runner_up=0.42, seconds=120.0),
        ], source="Team sync")
        entry = voiceprints.load_match_log()[0]
        assert entry.name == "Alice"
        assert entry.source == "Team sync"
        assert entry.score == pytest.approx(0.81)
        assert entry.runner_up == pytest.approx(0.42)
        assert entry.named


def test_rejections_are_recorded_too():
    """A threshold can only be judged against the scores it turned away."""
    with isolated_voiceprints():
        voiceprints.record_matches([
            voiceprints.Match(label="SPEAKER_00", score=0.40,
                              reason="closest match only 0.40"),
        ], source="Team sync")
        entry = voiceprints.load_match_log()[0]
        assert not entry.named
        assert "closest match" in entry.reason


def test_the_newest_attempt_comes_first():
    with isolated_voiceprints():
        voiceprints.record_matches([voiceprints.Match(label="A", name="First")])
        voiceprints.record_matches([voiceprints.Match(label="B", name="Second")])
        assert [e.name for e in voiceprints.load_match_log()] == ["Second", "First"]


def test_a_corrupt_line_does_not_lose_the_rest():
    with isolated_voiceprints():
        voiceprints.record_matches([voiceprints.Match(label="A", name="Alice")])
        with voiceprints.match_log_path().open("a", encoding="utf-8") as fh:
            fh.write("{ this is not json\n")
        assert [e.name for e in voiceprints.load_match_log()] == ["Alice"]


def test_a_missing_log_is_empty_not_an_error():
    with isolated_voiceprints():
        assert voiceprints.load_match_log() == []


# ---- the advice ------------------------------------------------------
def test_advice_waits_for_enough_evidence():
    assert "Not enough recordings" in voiceprints.suggest_threshold([])


def _logged(score: float, named: bool) -> voiceprints.LoggedMatch:
    return voiceprints.LoggedMatch(
        when=0.0, source="", label="SPEAKER_00",
        name="Alice" if named else "", score=score, runner_up=0.0,
        seconds=120.0, reason="" if named else f"closest match only {score:.2f}",
    )


def test_advice_names_the_gap_when_the_scores_separate_cleanly():
    entries = [_logged(0.9, True)] * 3 + [_logged(0.2, False)] * 3
    advice = voiceprints.suggest_threshold(entries)
    assert "Clean separation" in advice
    assert "0.20" in advice and "0.90" in advice


def test_advice_says_tuning_will_not_help_when_scores_overlap():
    """Accepted and rejected voices scoring alike is an enrolment problem, and
    telling someone to move the threshold would just lose real matches."""
    entries = [_logged(0.60, True)] * 3 + [_logged(0.58, False)] * 3
    advice = voiceprints.suggest_threshold(entries)
    assert "overlap" in advice
    assert "second enrolment" in advice


def test_advice_says_the_bar_is_too_high_when_nothing_matches():
    entries = [_logged(0.50, False)] * 6
    assert "bar is too high" in voiceprints.suggest_threshold(entries)


# ---- managing people -------------------------------------------------
def test_renaming_a_person_keeps_their_samples():
    with isolated_voiceprints():
        voiceprints.enroll("Alise", _vector(1, 0), seconds=90.0)
        voiceprints.rename_profile("Alise", "Alice")
        assert voiceprints.get_profile("Alise") is None
        assert len(voiceprints.get_profile("Alice").prints) == 1
        assert [p.name for p in voiceprints.load_profiles()] == ["Alice"]


def test_renaming_onto_someone_else_is_refused():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        voiceprints.enroll("Bob", _vector(0, 1), seconds=90.0)
        with pytest.raises(voiceprints.VoiceprintError, match="already enrolled"):
            voiceprints.rename_profile("Bob", "Alice")


def test_forgetting_one_bad_sample_keeps_the_person():
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0, source="good")
        voiceprints.enroll("Alice", _vector(0, 1), seconds=90.0, source="car")
        voiceprints.forget_sample("Alice", 1)
        remaining = voiceprints.get_profile("Alice")
        assert [p.source for p in remaining.prints] == ["good"]


def test_forgetting_the_last_sample_forgets_the_person():
    """A name with no captures would sit in the list matching nothing."""
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0)
        assert voiceprints.forget_sample("Alice", 0) is None
        assert voiceprints.load_profiles() == []


# ---- the dialog ------------------------------------------------------
def test_the_dialog_lists_enrolled_people(app):
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=90.0, source="Team sync")
        voiceprints.enroll("Bob", _vector(0, 1), seconds=90.0)
        dlg = SpeakersDialog(Settings())
        assert dlg.people.count() == 2
        assert "Alice" in dlg.people.item(0).text()
        assert "1 sample" in dlg.people.item(0).text()


def test_the_dialog_shows_where_a_sample_came_from(app):
    with isolated_voiceprints():
        voiceprints.enroll("Alice", _vector(1, 0), seconds=95.0, source="Team sync")
        dlg = SpeakersDialog(Settings())
        assert dlg.samples.item(0, 0).text() == "Team sync"
        assert dlg.samples.item(0, 1).text() == "1m 35s"


def test_the_empty_state_says_what_will_happen_instead(app):
    with isolated_voiceprints():
        dlg = SpeakersDialog(Settings())
        assert "Speaker 1, 2, 3" in dlg.voices_status.text()
        assert not dlg.delete_btn.isEnabled()


def test_saving_writes_the_thresholds_back(app, tmp_path, monkeypatch):
    from transcriber_studio import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "settings.json")
    with isolated_voiceprints():
        settings = Settings()
        dlg = SpeakersDialog(settings)
        dlg.threshold.setValue(0.62)
        dlg.margin.setValue(0.15)
        dlg.min_speech.setValue(20)
        dlg.min_enroll.setValue(45)
        dlg._save()
        assert settings.voiceprint_threshold == pytest.approx(0.62)
        assert settings.voiceprint_margin == pytest.approx(0.15)
        assert settings.voiceprint_min_speech_s == pytest.approx(20)
        assert settings.voiceprint_min_enroll_s == pytest.approx(45)
        assert config.load().voiceprint_threshold == pytest.approx(0.62)


def test_the_dialog_shows_the_recorded_attempts(app):
    with isolated_voiceprints():
        voiceprints.record_matches([
            voiceprints.Match(label="SPEAKER_00", name="Alice", score=0.8,
                              runner_up=0.3, seconds=120.0),
        ], source="Team sync")
        dlg = SpeakersDialog(Settings())
        assert dlg.log.rowCount() == 1
        assert dlg.log.item(0, 1).text() == "Team sync"
        assert dlg.log.item(0, 3).text() == "Alice"
        assert dlg.log.item(0, 4).text() == "0.80"


def test_a_rejection_shows_its_reason_rather_than_a_blank(app):
    with isolated_voiceprints():
        voiceprints.record_matches([
            voiceprints.Match(label="SPEAKER_00", score=0.4,
                              reason="closest match only 0.40", seconds=120.0),
        ], source="Team sync")
        dlg = SpeakersDialog(Settings())
        assert dlg.log.item(0, 3).text() == "closest match only 0.40"
