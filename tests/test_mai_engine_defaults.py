# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choosing MAI-Transcribe turns off the one stage it makes redundant.

Denoise re-encodes to uncompressed WAV, which took a real 70 MB recording to
140 MB before upload — paid for in local CPU and then again in upload time —
and MAI states noise robustness as one of its own features.

AI Cleanup is deliberately left alone. It was switched off here too at first,
on the theory that the model's "clean" style made it redundant; measured, that
style deletes real clauses, so the recommended setup is verbatim plus Cleanup.

The part worth testing is not that the boxes clear: it is that the user is told,
once, and never nagged about a change that did not happen.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from transcriber_studio.config import Settings
from transcriber_studio.transcriber import (
    ENGINE_ELEVENLABS,
    ENGINE_GEMINI,
    ENGINE_LOCAL,
    ENGINE_MAI,
)
from transcriber_studio.ui.options_panel import OptionsPanel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    """Every dialog the panel raises, instead of raising it."""
    seen: list[dict] = []
    monkeypatch.setattr(
        QMessageBox, "exec",
        lambda self: seen.append({
            "title": self.windowTitle(),
            "text": self.text(),
            "info": self.informativeText(),
        }) or 0,
    )
    return seen


def _panel(**overrides) -> OptionsPanel:
    settings = Settings()
    settings.denoise_enabled = overrides.get("denoise", True)
    settings.ai_cleanup_enabled = overrides.get("cleanup", True)
    settings.stt_engine = overrides.get("engine", ENGINE_LOCAL)
    return OptionsPanel(settings)


def _choose(panel: OptionsPanel, engine: str) -> None:
    panel.engine.setCurrentIndex(panel.engine.findData(engine))


# ---- the switch ------------------------------------------------------
def test_choosing_mai_turns_denoise_off_and_leaves_cleanup_alone(app, dialogs):
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    assert not panel.denoise_on.isChecked()
    assert panel.ai_cleanup_on.isChecked(), "cleanup must survive the switch"


def test_the_user_is_told_what_changed(app, dialogs):
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    assert len(dialogs) == 1
    info = dialogs[0]["info"]
    assert "Denoising is off" in info
    assert "AI Cleanup" not in info
    assert "140 MB" in info          # the reason, not just the fact


def test_the_note_says_it_can_be_undone(app, dialogs):
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    assert "Turn it back on" in dialogs[0]["info"]


# ---- when it should stay quiet ---------------------------------------
def test_nothing_is_said_when_denoise_was_already_off(app, dialogs):
    """Nothing happened, so there is nothing to announce."""
    panel = _panel(denoise=False, cleanup=True)
    _choose(panel, ENGINE_MAI)
    assert dialogs == []


def test_cleanup_being_on_does_not_trigger_the_dialog_by_itself(app, dialogs):
    panel = _panel(denoise=False, cleanup=True)
    _choose(panel, ENGINE_MAI)
    assert dialogs == []
    assert panel.ai_cleanup_on.isChecked()


def test_switching_to_mai_again_does_not_nag(app, dialogs):
    """It fires on the choice, not on every run — and the second choice
    changes nothing, because they are already off."""
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    _choose(panel, ENGINE_LOCAL)
    _choose(panel, ENGINE_MAI)
    assert len(dialogs) == 1


def test_turning_them_back_on_sticks_until_the_engine_is_chosen_again(app, dialogs):
    """The switch is a default, not a rule: re-enabling must hold."""
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    panel.denoise_on.setChecked(True)
    assert panel.denoise_on.isChecked()


@pytest.mark.parametrize("engine", [ENGINE_LOCAL, ENGINE_ELEVENLABS, ENGINE_GEMINI])
def test_the_other_engines_are_left_alone(app, dialogs, engine):
    panel = _panel()
    _choose(panel, engine)
    assert panel.denoise_on.isChecked()
    assert panel.ai_cleanup_on.isChecked()
    assert dialogs == []


def test_starting_up_already_on_mai_does_not_pop_a_dialog(app, dialogs):
    """Building the panel is not the user choosing anything."""
    _panel(engine=ENGINE_MAI)
    assert dialogs == []


# ---- it reaches the job ----------------------------------------------
def test_the_settings_the_panel_writes_have_both_off(app, dialogs):
    panel = _panel()
    _choose(panel, ENGINE_MAI)
    settings = Settings()
    panel.apply_to(settings)
    assert settings.denoise_enabled is False
    assert settings.ai_cleanup_enabled is True
    assert settings.stt_engine == ENGINE_MAI
