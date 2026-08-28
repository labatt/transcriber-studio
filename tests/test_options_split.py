# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Options column and the Settings dialog own different settings.

The column writes back on every run. Anything it does not own must survive
that untouched, or a value set in Settings would be silently reverted by a
widget the panel no longer even shows.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from transcriber_studio.config import Settings
from transcriber_studio.ui.options_panel import OptionsPanel
from transcriber_studio.ui.settings_dialog import SettingsDialog

#: Deliberately unlike the defaults, so an accidental overwrite is obvious.
TUNING = dict(
    vad_threshold=0.75,
    vad_min_silence_ms=1234,
    vad_speech_pad_ms=321,
    vad_min_speech_ms=222,
    vad_max_speech_s=45.0,
    bias_extra_terms="Kenobi, Grievous",
    bias_max_chars=1700,
    hallucination_guard=False,
    channel_mode="per_channel",
    channel_names="Interviewer, Guest",
    line_mode="wrap",
    wrap_chars=123,
    include_timestamps=True,
    newline="lf",
    filename_template="{date}__{name}",
    overwrite=True,
    sanitize_names=False,
    owner_names="Chris",
    glossary_model="gemini-flash-latest",
    glossary_temperature=0.4,
    glossary_chunk_token_threshold=54321,
    force_reextract=True,
    prompt_cache_enabled=False,
)


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


def test_the_options_column_leaves_the_tuning_alone():
    settings = Settings(**TUNING)
    OptionsPanel(settings).apply_to(settings)

    for name, expected in TUNING.items():
        assert getattr(settings, name) == expected, f"the panel overwrote {name}"


def test_the_settings_dialog_round_trips_every_setting_that_moved():
    settings = Settings(**TUNING)
    dialog = SettingsDialog(settings)
    dialog._accept()

    for name, expected in TUNING.items():
        assert getattr(settings, name) == expected, f"the dialog lost {name}"


def test_the_options_column_still_owns_the_per_job_choices():
    settings = Settings()
    panel = OptionsPanel(settings)

    panel.denoise_on.setChecked(False)
    panel.vad_on.setChecked(False)
    panel.bias_on.setChecked(False)
    panel.include_speakers.setChecked(False)
    panel.apply_to(settings)

    assert settings.denoise_enabled is False
    assert settings.vad_enabled is False
    assert settings.bias_enabled is False
    assert settings.include_speakers is False


def test_the_column_is_no_longer_a_wall_of_controls():
    """The complaint that started this: too much in one scrolling strip."""
    from PySide6.QtWidgets import QGroupBox

    panel = OptionsPanel(Settings())
    boxes = panel.findChildren(QGroupBox)

    assert len(boxes) <= 5, [b.title() for b in boxes]
    titles = {b.title() for b in boxes}
    assert "Line formatting" not in titles, "line formatting belongs in Settings now"


def test_settings_is_tabbed_rather_than_one_long_scroll():
    dialog = SettingsDialog(Settings())

    assert dialog.tabs.count() >= 5
    titles = {dialog.tabs.tabText(i) for i in range(dialog.tabs.count())}
    assert {"Engines", "Audio", "Output", "AI Cleanup"} <= titles


def test_opening_and_saving_settings_changes_nothing_by_itself():
    """A spin box whose range excludes the default silently rewrites it.

    vad_max_speech_s defaults to 0.0 meaning "no cap"; a minimum of 1.0 turned
    that into a one-second cap just by opening the dialog and pressing Save.
    """
    defaults = Settings()
    settings = Settings()
    dialog = SettingsDialog(settings)
    dialog._accept()

    changed = {
        name: (getattr(defaults, name), getattr(settings, name))
        for name in vars(defaults)
        if getattr(defaults, name) != getattr(settings, name)
    }
    assert not changed, f"opening Settings and saving altered: {changed}"
