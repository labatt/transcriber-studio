# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The glossary pickers, the merge/review flow, and the default cleanup model."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # tests run without a display

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from tests.support import isolated_glossary_dir
from transcriber_studio import glossary_store
from transcriber_studio.config import Settings, cleanup_model, cleanup_provider
from transcriber_studio.glossary_merge import Part
from transcriber_studio.glossary_store import CONFLICT_KEY
from transcriber_studio.ui.glossary_dialog import (
    NEW_GLOSSARY,
    PER_RECORDING_LABEL,
    CombineGlossariesDialog,
    GlossaryLibraryDialog,
    populate_glossary_combo,
)
from transcriber_studio.ui.options_panel import OptionsPanel

_app = QApplication.instance() or QApplication([])


def _type_term(table, *values: str) -> int:
    """Add a row and fill its editable cells, as a person would."""
    table.add_row()
    row = table.table.rowCount() - 1
    for col, value in enumerate(values):
        table.table.item(row, col).setText(value)
    return row


def _row_of(table, term: str) -> int:
    return next(
        r for r in range(table.rowCount()) if table.item(r, 0).text() == term
    )


def _clashing_pair():
    """Two glossaries that agree on one term and disagree about another."""
    left = glossary_store.create(
        "Acme",
        terms=[
            {"canonical": "Scribe", "variants": ["scribes"], "type": "product"},
            {"canonical": "GrowthMark", "variants": [], "type": "product"},
        ],
    )
    right = glossary_store.create(
        "Vendor",
        terms=[
            {"canonical": "Scribe", "variants": ["Scribe AI"], "type": "concept"},
            {"canonical": "Cadence", "variants": [], "type": "concept"},
        ],
    )
    return left, right


# ---- choosing a glossary ----------------------------------------------


def test_combo_lists_the_library_with_per_recording_first():
    with isolated_glossary_dir():
        acme = glossary_store.create("Acme Account")
        glossary_store.create("Internal Standups")
        combo = QComboBox()

        populate_glossary_combo(combo, acme.id)

        assert combo.itemText(0) == PER_RECORDING_LABEL
        assert combo.itemData(0) == ""
        assert combo.currentData() == acme.id
        assert combo.count() == 3


def test_a_deleted_glossary_stays_visible_as_missing():
    """Losing the vocabulary silently is worse than being told it is gone."""
    with isolated_glossary_dir():
        combo = QComboBox()

        populate_glossary_combo(combo, "deleted-one")

        assert combo.currentData() == "deleted-one"
        assert "missing" in combo.currentText()


def test_options_panel_round_trips_the_shared_glossary_choice():
    with isolated_glossary_dir():
        acme = glossary_store.create("Acme Account")
        settings = Settings(ai_cleanup_enabled=True)
        panel = OptionsPanel(settings)

        panel.shared_glossary.setCurrentIndex(panel.shared_glossary.findData(acme.id))
        panel.apply_to(settings)

        assert settings.glossary_shared_id == acme.id

        # …and a panel rebuilt from those settings comes up on the same one.
        assert OptionsPanel(settings).shared_glossary.currentData() == acme.id


# ---- editing a glossary -----------------------------------------------


def test_library_dialog_creates_edits_and_saves_terms():
    with isolated_glossary_dir():
        created = glossary_store.create("Acme Account")
        dlg = GlossaryLibraryDialog(selected=created.id)

        _type_term(dlg.terms, "GrowthMark", "product", "growth mark, growth market")
        dlg._save_current()

        stored = glossary_store.load(created.id)
        assert [t["canonical"] for t in stored.terms] == ["GrowthMark"]
        assert stored.terms[0]["variants"] == ["growth mark", "growth market"]
        assert stored.terms[0]["type"] == "product"


def test_renaming_keeps_the_contents_and_the_id():
    with isolated_glossary_dir():
        created = glossary_store.create(
            "Acme", terms=[{"canonical": "Scribe", "variants": [], "type": "product"}]
        )
        dlg = GlossaryLibraryDialog(selected=created.id)

        dlg._current.name = "Acme Corp"       # what the rename prompt sets
        glossary_store.save(dlg._current)
        dlg._reload_list(select=created.id)

        stored = glossary_store.load(created.id)
        assert stored.name == "Acme Corp"
        assert [t["canonical"] for t in stored.terms] == ["Scribe"]
        assert [g.name for g in glossary_store.list_glossaries()] == ["Acme Corp"]


# ---- merging, and the review it produces ------------------------------


def test_merging_into_the_open_glossary_dedupes_and_tags_the_clash():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)

        result = dlg._merge_into(
            glossary_store.load(left.id), [Part.of(left), Part.of(right)]
        )

        stored = glossary_store.load(left.id)
        by_name = {t["canonical"]: t for t in stored.terms}
        assert sorted(by_name) == ["Cadence", "GrowthMark", "Scribe"]   # deduped
        assert by_name["Scribe"]["variants"] == ["Scribe AI", "scribes"]
        assert by_name["Scribe"][CONFLICT_KEY]["field"] == "type"
        assert CONFLICT_KEY not in by_name["GrowthMark"]
        assert result.new_conflicts == 1
        # The source glossary is left alone.
        assert len(glossary_store.load(right.id).terms) == 2


def test_a_merge_opens_the_view_filtered_to_the_rows_needing_review():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)

        dlg._merge_into(glossary_store.load(left.id), [Part.of(left), Part.of(right)])

        assert dlg.terms.only_conflicts.isChecked()
        assert dlg.tabs.currentWidget() is dlg.terms
        table = dlg.terms.table
        visible = [
            table.item(r, 0).text()
            for r in range(table.rowCount())
            if not table.isRowHidden(r)
        ]
        assert visible == ["Scribe"]
        assert "1 row(s) need review" in dlg.terms.count_label.text()


def test_editing_the_column_they_disagreed_about_clears_the_tag():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)
        dlg._merge_into(glossary_store.load(left.id), [Part.of(left), Part.of(right)])

        table = dlg.terms.table
        table.item(_row_of(table, "Scribe"), 1).setText("concept")   # settled
        dlg._save_current()

        stored = {t["canonical"]: t for t in glossary_store.load(left.id).terms}
        assert stored["Scribe"]["type"] == "concept"
        assert CONFLICT_KEY not in stored["Scribe"]
        assert dlg.terms.conflict_count() == 0


def test_keep_as_is_drops_the_tag_without_changing_the_row():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)
        dlg._merge_into(glossary_store.load(left.id), [Part.of(left), Part.of(right)])

        table = dlg.terms.table
        row = _row_of(table, "Scribe")
        before = table.item(row, 1).text()
        table.selectRow(row)
        dlg.terms.clear_selected_tags()
        dlg._save_current()

        stored = {t["canonical"]: t for t in glossary_store.load(left.id).terms}
        assert stored["Scribe"]["type"] == before
        assert CONFLICT_KEY not in stored["Scribe"]


def test_deleting_the_row_is_the_other_way_out():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)
        dlg._merge_into(glossary_store.load(left.id), [Part.of(left), Part.of(right)])

        table = dlg.terms.table
        table.selectRow(_row_of(table, "Scribe"))
        dlg.terms.remove_selected()
        dlg._save_current()

        stored = glossary_store.load(left.id)
        assert "Scribe" not in [t["canonical"] for t in stored.terms]
        assert stored.conflict_count() == 0


def test_a_glossary_with_open_conflicts_says_so_in_the_library_list():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = GlossaryLibraryDialog(selected=left.id)

        dlg._merge_into(glossary_store.load(left.id), [Part.of(left), Part.of(right)])

        assert "to resolve" in glossary_store.load(left.id).summary()
        assert any(
            "to resolve" in dlg.list.item(i).text() for i in range(dlg.list.count())
        )


def test_combine_dialog_defaults_to_the_open_glossary():
    with isolated_glossary_dir():
        left, right = _clashing_pair()
        dlg = CombineGlossariesDialog(current_id=left.id)

        sources, destination, _name = dlg.selection()
        assert sources == [left.id]
        assert destination == left.id

        # Picking "a new glossary" hands back an empty destination and a name.
        dlg.destination.setCurrentIndex(dlg.destination.findData(NEW_GLOSSARY))
        dlg.new_name.setText("Everything")
        for i in range(dlg.sources.count()):
            dlg.sources.item(i).setCheckState(Qt.CheckState.Checked)
        sources, destination, name = dlg.selection()
        assert sorted(sources) == sorted([left.id, right.id])
        assert destination == "" and name == "Everything"


# ---- the default cleanup model ----------------------------------------


def test_default_model_fills_in_when_a_run_has_not_picked_one():
    settings = Settings(ai_default_provider="anthropic", ai_default_model="claude-opus-5")

    assert cleanup_provider(settings) == "anthropic"
    assert cleanup_model(settings) == "claude-opus-5"

    # A pick made for the run at hand still wins.
    settings.ai_cleanup_provider = "openai"
    settings.ai_cleanup_model = "gpt-5"
    assert cleanup_provider(settings) == "openai"
    assert cleanup_model(settings) == "gpt-5"
