# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renaming from the recordings list.

The behaviour that matters is the order: the name is kept locally whatever
happens to the push, and a refresh shows the name the user chose rather than
the one Plaud still has.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from transcriber_studio import name_store
from transcriber_studio.config import Settings
from transcriber_studio.models import Recording, Source
from transcriber_studio.ui.recordings_tab import NAME_COL, RecordingsTab

from .support import isolated_name_store


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _recording(rec_id: str = "f1", name: str = "REC_001") -> Recording:
    return Recording(source=Source.PLAUD, id=rec_id, name=name, date="2026-09-01")


def _tab(settings: Settings | None = None) -> RecordingsTab:
    tab = RecordingsTab(settings or Settings())
    tab._on_loaded([_recording()])
    return tab


def test_the_name_cell_is_editable_and_the_others_are_not(app):
    with isolated_name_store():
        tab = _tab()
        assert tab.table.item(0, NAME_COL).flags() & Qt.ItemFlag.ItemIsEditable
        for col in (2, 3, 4, 5):
            assert not (tab.table.item(0, col).flags() & Qt.ItemFlag.ItemIsEditable)


def test_editing_the_name_records_it_locally(app):
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("Weekly sync")
        entry = name_store.get("f1")
        assert entry is not None
        assert entry.name == "Weekly sync"
        assert entry.original == "REC_001"


def test_without_a_token_nothing_is_pushed(app):
    """Push is opt-in; the rename still has to work without it."""
    with isolated_name_store():
        settings = Settings()
        settings.plaud_rename_push = True
        settings.plaud_web_token = ""       # opted in but never pasted one
        tab = _tab(settings)
        tab.table.item(0, NAME_COL).setText("Weekly sync")
        assert name_store.get("f1").name == "Weekly sync"
        assert not tab._rename_workers, "should not have tried to reach Plaud"


def test_an_empty_name_is_put_back(app):
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("   ")
        assert tab.table.item(0, NAME_COL).text() == "REC_001"
        assert name_store.get("f1") is None


def test_renaming_to_the_same_name_records_nothing(app):
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("REC_001")
        assert name_store.get("f1") is None


def test_a_reload_shows_the_local_name_not_plaud_s(app):
    """Plaud keeps sending the old name until the push lands — and after a
    failed push it keeps sending it for good."""
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("Weekly sync")
        tab._on_loaded([_recording()])      # a Refresh, straight from Plaud
        assert tab.table.item(0, NAME_COL).text() == "Weekly sync"
        assert tab._recordings[0].name == "Weekly sync"


def test_loading_the_list_does_not_look_like_an_edit(app):
    """Populating the table writes into the Name column too."""
    with isolated_name_store():
        tab = _tab()
        tab._on_loaded([_recording(), _recording("f2", "REC_002")])
        assert name_store.load(force=True) == {}


def test_an_unsynced_name_says_so_in_the_tooltip(app):
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("Weekly sync")
        tooltip = tab.table.item(0, NAME_COL).toolTip()
        assert "not on Plaud yet" in tooltip
        assert "REC_001" in tooltip


def test_a_synced_name_says_plaud_has_it(app):
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=True)
        tab = _tab()
        tooltip = tab.table.item(0, NAME_COL).toolTip()
        assert "Plaud has this name too" in tooltip


def test_the_new_name_reaches_the_job_and_so_the_filename(app):
    """{name} in the filename template is the recording's name."""
    with isolated_name_store():
        tab = _tab()
        tab.table.item(0, NAME_COL).setText("Weekly sync")
        tab.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
        selected = tab.selected()
        assert [r.name for r in selected] == ["Weekly sync"]
