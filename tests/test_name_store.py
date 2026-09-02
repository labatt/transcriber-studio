# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Names given to Plaud recordings survive a push that fails.

The rename is committed locally and pushed second, so the interesting cases are
all about what happens when the second half does not work.
"""

from __future__ import annotations

from transcriber_studio import name_store

from .support import isolated_name_store


def test_a_name_survives_a_failed_push():
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=False)
        entry = name_store.get("f1")
        assert entry is not None
        assert entry.name == "Weekly sync"
        assert entry.pushed is False
        assert name_store.unsynced() == {"f1": entry}


def test_a_pushed_name_is_not_listed_as_unsynced():
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=True)
        assert name_store.unsynced() == {}


def test_a_later_push_clears_the_unsynced_mark():
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=False)
        name_store.mark_pushed("f1")
        assert name_store.get("f1").pushed is True
        assert name_store.unsynced() == {}


def test_renaming_twice_still_remembers_what_plaud_called_it():
    """Otherwise the second rename records the first one as the original."""
    with isolated_name_store():
        name_store.record("f1", "First try", original="REC_001", pushed=True)
        name_store.record("f1", "Second try", original="First try", pushed=True)
        entry = name_store.get("f1")
        assert entry.name == "Second try"
        assert entry.original == "REC_001"


def test_a_local_name_outranks_the_one_plaud_sends():
    with isolated_name_store():
        assert name_store.name_for("f1", "REC_001") == "REC_001"
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=False)
        assert name_store.name_for("f1", "REC_001") == "Weekly sync"


def test_names_survive_a_reload_from_disk():
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=False)
        name_store.load(force=True)     # as a restart would
        assert name_store.get("f1").name == "Weekly sync"


def test_forgetting_a_name_falls_back_to_plaud():
    with isolated_name_store():
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=True)
        name_store.forget("f1")
        assert name_store.get("f1") is None
        assert name_store.name_for("f1", "REC_001") == "REC_001"


def test_a_corrupt_store_is_not_fatal():
    """These names are a convenience; losing them must not stop the app."""
    with isolated_name_store() as path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json", encoding="utf-8")
        assert name_store.load(force=True) == {}
        name_store.record("f1", "Weekly sync", original="REC_001", pushed=False)
        assert name_store.get("f1").name == "Weekly sync"


def test_entries_without_a_name_are_ignored():
    with isolated_name_store() as path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"f1": {"name": "", "original": "x"}, "f2": {"name": "Kept"}}',
            encoding="utf-8",
        )
        entries = name_store.load(force=True)
        assert set(entries) == {"f2"}
