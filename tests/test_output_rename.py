# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renaming the files a job wrote.

A job usually writes the same transcript several times — .txt, .srt, .json —
under one stem, so they move together and only the extensions differ
afterwards. The failure that matters is a half-renamed set: some files under
the old name and some under the new one is worse than either.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from transcriber_studio.jobs import JobResult
from transcriber_studio.models import Recording, Source
from transcriber_studio.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    """A main window with one finished job whose files are really on disk."""
    from transcriber_studio import queue_store

    monkeypatch.setattr(queue_store, "QUEUE_PATH", tmp_path / "queue.json", raising=False)
    win = MainWindow()
    win._results[0] = JobResult(
        recording=Recording(source=Source.PLAUD, id="f1", name="Team sync"),
    )
    return win


def _write_set(directory: Path, stem: str, suffixes=(".txt", ".srt", ".json")) -> list[str]:
    paths = []
    for suffix in suffixes:
        path = directory / f"{stem}{suffix}"
        path.write_text(f"contents of {path.name}", encoding="utf-8")
        paths.append(str(path))
    return paths


def _answer(monkeypatch, text: str, ok: bool = True):
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (text, ok))


def _silence_warnings(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda _parent, _title, text, *a, **k: seen.append(text),
    )
    return seen


# ---- the ordinary case -----------------------------------------------
def test_the_whole_set_is_renamed_together(window, monkeypatch, tmp_path):
    window._results[0].output_paths = _write_set(tmp_path, "2026-09-01_Team sync")
    _answer(monkeypatch, "Board review")
    window._rename_output(0)

    on_disk = sorted(p.name for p in tmp_path.iterdir())
    assert on_disk == ["Board review.json", "Board review.srt", "Board review.txt"]


def test_the_job_remembers_where_the_files_went(window, monkeypatch, tmp_path):
    window._results[0].output_paths = _write_set(tmp_path, "old")
    _answer(monkeypatch, "new")
    window._rename_output(0)
    assert [Path(p).name for p in window._results[0].output_paths] == [
        "new.txt", "new.srt", "new.json",
    ]
    assert all(Path(p).is_file() for p in window._results[0].output_paths)


def test_the_contents_are_untouched(window, monkeypatch, tmp_path):
    window._results[0].output_paths = _write_set(tmp_path, "old", (".txt",))
    _answer(monkeypatch, "new")
    window._rename_output(0)
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "contents of old.txt"


# ---- refusals --------------------------------------------------------
def test_cancelling_changes_nothing(window, monkeypatch, tmp_path):
    window._results[0].output_paths = _write_set(tmp_path, "old")
    _answer(monkeypatch, "new", ok=False)
    window._rename_output(0)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "old.json", "old.srt", "old.txt",
    ]


def test_an_empty_name_is_refused(window, monkeypatch, tmp_path):
    """Not renamed to "untitled" — sanitise() falls back to that so generated
    filenames always exist, and it must not stand in for a name the user
    cleared on purpose."""
    window._results[0].output_paths = _write_set(tmp_path, "old", (".txt",))
    warnings = _silence_warnings(monkeypatch)
    _answer(monkeypatch, "   ")
    window._rename_output(0)
    assert (tmp_path / "old.txt").exists()
    assert not (tmp_path / "untitled.txt").exists()
    assert any("needs a name" in w for w in warnings)


def test_the_same_name_is_a_no_op(window, monkeypatch, tmp_path):
    window._results[0].output_paths = _write_set(tmp_path, "old", (".txt",))
    _answer(monkeypatch, "old")
    window._rename_output(0)
    assert (tmp_path / "old.txt").exists()


def test_renaming_onto_an_existing_file_is_refused(window, monkeypatch, tmp_path):
    """That file was written by something else, and this must not destroy it."""
    window._results[0].output_paths = _write_set(tmp_path, "old", (".txt",))
    (tmp_path / "taken.txt").write_text("someone else's work", encoding="utf-8")
    warnings = _silence_warnings(monkeypatch)
    _answer(monkeypatch, "taken")
    window._rename_output(0)

    assert (tmp_path / "old.txt").exists()
    assert (tmp_path / "taken.txt").read_text(encoding="utf-8") == "someone else's work"
    assert any("Already there" in w for w in warnings)


def test_a_clash_on_one_format_blocks_the_whole_set(window, monkeypatch, tmp_path):
    """Renaming the rest anyway would split the set across two names."""
    window._results[0].output_paths = _write_set(tmp_path, "old")
    (tmp_path / "new.srt").write_text("in the way", encoding="utf-8")
    _silence_warnings(monkeypatch)
    _answer(monkeypatch, "new")
    window._rename_output(0)
    assert (tmp_path / "old.txt").exists()
    assert (tmp_path / "old.json").exists()


def test_files_that_are_gone_are_reported_not_crashed_on(window, monkeypatch, tmp_path):
    window._results[0].output_paths = [str(tmp_path / "never-written.txt")]
    warnings = _silence_warnings(monkeypatch)
    window._rename_output(0)
    assert any("on disk any more" in w for w in warnings)


def test_a_job_with_no_output_is_handled(window, monkeypatch):
    warnings = _silence_warnings(monkeypatch)
    window._rename_output(0)
    assert warnings


# ---- partial sets ----------------------------------------------------
def test_a_missing_sibling_does_not_stop_the_rest(window, monkeypatch, tmp_path):
    paths = _write_set(tmp_path, "old", (".txt", ".srt"))
    paths.append(str(tmp_path / "old.json"))     # recorded, never written
    window._results[0].output_paths = paths
    _answer(monkeypatch, "new")
    window._rename_output(0)
    assert (tmp_path / "new.txt").exists()
    assert (tmp_path / "new.srt").exists()


def test_the_prompt_says_how_many_are_missing(window, monkeypatch, tmp_path):
    paths = _write_set(tmp_path, "old", (".txt",))
    paths.append(str(tmp_path / "old.json"))
    window._results[0].output_paths = paths
    seen: dict[str, str] = {}

    def fake(_parent, _title, prompt, *a, **k):
        seen["prompt"] = prompt
        return ("new", True)

    monkeypatch.setattr(QInputDialog, "getText", fake)
    window._rename_output(0)
    assert "1 more" in seen["prompt"]
    assert "no longer on disk" in seen["prompt"]


def test_invalid_characters_are_cleaned_before_writing(window, monkeypatch, tmp_path):
    """Windows will not take them, and the app already sanitises what it writes."""
    window._results[0].output_paths = _write_set(tmp_path, "old", (".txt",))
    window.settings.sanitize_names = True
    _answer(monkeypatch, 'bad:name?here')
    window._rename_output(0)
    written = [p.name for p in tmp_path.iterdir()]
    assert written == ["badnamehere.txt"]
