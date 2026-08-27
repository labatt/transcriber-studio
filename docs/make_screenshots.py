#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the README screenshots from mocked-up data.

    python docs/make_screenshots.py

Two rules this script exists to enforce:

* **Never the real app directory.** It points APP_DIR, the glossary library and
  the caches at a throwaway folder before anything is imported that might read
  them. A screenshot taken against a real install puts real meeting names and
  real client vocabulary into a public README.
* **Real fonts.** Qt's offscreen platform has no font database, so every glyph
  renders as a missing-glyph box. These are grabbed from actual widgets on the
  normal platform, which means windows flicker on screen while it runs.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SANDBOX = Path(tempfile.mkdtemp(prefix="ts_shots_"))
OUT = ROOT / "docs" / "screenshots"

from transcriber_studio import config  # noqa: E402  (must precede anything using APP_DIR)

config.APP_DIR = SANDBOX
config.CONFIG_PATH = SANDBOX / "settings.json"
config.LEGACY_APP_DIRS = ()

from PySide6.QtWidgets import QApplication, QGroupBox  # noqa: E402

from transcriber_studio import (  # noqa: E402  # noqa: E402
    audio_cache,
    denoise,
    glossary_store,
    history,
    queue_store,
    resume,
)

for module, attr, value in (
    (glossary_store, "GLOSSARY_DIR", SANDBOX / "glossaries"),
    (denoise, "CACHE_DIR", SANDBOX / "denoise_cache"),
    (audio_cache, "CACHE_DIR", SANDBOX / "audio_cache"),
    (resume, "RESUME_DIR", SANDBOX / "resume"),
    (history, "HISTORY_PATH", SANDBOX / "history.json"),
    (queue_store, "QUEUE_PATH", SANDBOX / "queue.json"),
):
    setattr(module, attr, value)

from transcriber_studio.config import Settings  # noqa: E402
from transcriber_studio.glossary_merge import Part  # noqa: E402
from transcriber_studio.jobs import JobResult  # noqa: E402
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult  # noqa: E402
from transcriber_studio.ui.ai_cleanup_dialog import AICleanupDialog  # noqa: E402
from transcriber_studio.ui.glossary_dialog import GlossaryLibraryDialog  # noqa: E402
from transcriber_studio.ui.install_help import InstallHelpDialog  # noqa: E402
from transcriber_studio.ui.main_window import MainWindow  # noqa: E402
from transcriber_studio.ui.options_panel import OptionsPanel  # noqa: E402
from transcriber_studio.ui.settings_dialog import SettingsDialog  # noqa: E402

app = QApplication(sys.argv)


def shot(widget, name: str, width: int | None = None, height: int | None = None) -> None:
    if width and height:
        widget.resize(width, height)
    widget.show()
    for _ in range(6):
        app.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    widget.grab().save(str(path))
    print(f"  {path.relative_to(ROOT)}  ({widget.width()}x{widget.height()})")
    widget.hide()


# --- invented content, so nothing real is ever in a screenshot -------------
def demo_settings() -> Settings:
    s = Settings()
    s.output_dir = str(Path.home() / "Documents" / "Transcripts")
    s.ai_cleanup_enabled = True
    s.ai_cleanup_provider = "anthropic"
    s.ai_cleanup_model = "claude-sonnet-5"
    s.ai_default_provider = "anthropic"
    s.ai_default_model = "claude-sonnet-5"
    s.denoise_enabled = True
    s.deep_filter_path = r"C:\Tools\deep-filter.exe"
    s.bias_extra_terms = "NorthGate, Meridian, Okonkwo"
    s.owner_names = "Alex Rivera, Alex R, Alex"
    s.formats = ["txt", "srt", "md"]
    return s


def demo_glossaries():
    """A library with a genuine merge conflict left in it to review."""
    acme = glossary_store.create(
        "Acme Account",
        terms=[
            {"canonical": "NorthGate", "variants": ["north gate", "northgate"], "type": "product"},
            {"canonical": "Meridian", "variants": ["meridien"], "type": "product"},
            {"canonical": "Dana Okonkwo", "variants": ["Dana O"], "type": "person"},
        ],
    )
    vendor = glossary_store.create(
        "Vendor calls",
        terms=[
            {"canonical": "NorthGate", "variants": ["North Gate"], "type": "concept"},
            {"canonical": "Cadence", "variants": [], "type": "concept"},
        ],
    )
    glossary_store.create("Board meetings")
    return acme, vendor


def demo_recordings() -> list[Recording]:
    """A cloud library worth looking at, none of it real."""
    return [
        Recording(Source.PLAUD, "demo-a", "Acme quarterly review", date="2026-08-26",
                  duration="42m10s"),
        Recording(Source.PLAUD, "demo-b", "Vendor call — Meridian rollout",
                  date="2026-08-25", duration="18m03s"),
        Recording(Source.PLAUD, "demo-c", "Okonkwo 1:1", date="2026-08-25", duration="27m31s"),
        Recording(Source.PLAUD, "demo-d", "Board prep", date="2026-08-24", duration="51m18s"),
        Recording(Source.PLAUD, "demo-e", "NorthGate handover", date="2026-08-22",
                  duration="1h04m"),
    ]


def demo_jobs(window: MainWindow) -> None:
    """A queue that looks like a normal afternoon's work."""
    rows = [
        ("Acme quarterly review", "Done · 3 speakers", 100),
        ("Vendor call — Meridian rollout", "Done · 2 speakers", 100),
        ("Standup 2026-08-26", "AI Cleanup…", 62),
    ]
    recordings = []
    for i, (name, _status, _pct) in enumerate(rows):
        rec = Recording(
            source=Source.PLAUD if i < 2 else Source.LOCAL,
            id=f"demo-{i}",
            name=name,
            date="2026-08-26",
            duration=("42m10s", "18m03s", "9m44s")[i],
            local_path=None if i < 2 else r"C:\Audio\standup.m4a",
        )
        recordings.append(rec)
    window._append_queue_rows(recordings, 0)
    for i, (name, status, pct) in enumerate(rows):
        transcript = TranscriptResult(
            recording=recordings[i],
            segments=[Segment(0.0, 4.0, "…", "Alex Rivera")],
            language="en",
            model="large-v3",
            speakers=["Alex Rivera", "Dana Okonkwo"][: (3, 2, 2)[i] - 1] or ["Alex Rivera"],
        )
        window._results[i] = JobResult(
            recordings[i],
            output_paths=[str(Path.home() / "Documents" / "Transcripts" / f"{name}.txt")],
            transcript=transcript,
        )
        window._set_status(i, status)
        bar = window._progress_bar_at(i)
        if bar is not None:
            bar.setValue(pct)
    window._log("Denoise: DeepFilterNet (deep-filter binary) — 42.2 min of audio…")
    window._log("VAD: kept 31.8 min of 42.2 min — 10.4 min of non-speech never reached the decoder.")
    window._log("Vocabulary biasing: 14 term(s), 173 chars (NorthGate, Meridian, Dana Okonkwo…)")
    window._log("AI Cleanup: glossary ready — 3 speaker(s), 14 term(s)")


def stub_live_calls() -> None:
    """Keep the screenshots off the network and out of a real account.

    MainWindow refreshes the PLAUD account on construction and opens the
    first-run wizard on a timer. Left alone, the first blocks on a CLI call and
    would print a real account name into the image, and the second opens a modal
    dialog that never returns.
    """
    from transcriber_studio.ui import main_window as mw

    mw.MainWindow._refresh_account = lambda self: self.account_label.setText(
        "✓ demo  (alex@example.com)"
    )
    mw.MainWindow._maybe_run_setup_wizard = lambda self: None

    # Find ffmpeg wherever it actually is, so the pipeline does not report
    # itself as unavailable in the marquee screenshot.
    from transcriber_studio import components

    components.refresh_path()

    # Show a denoiser path that is not somebody's home directory. This is the
    # one deliberate fiction in these images: the binary really is installed,
    # its real path just has a username in it.
    denoise.binary_path = lambda configured="": (configured or None)

    # Do not list whatever models this machine happens to have. A screenshot
    # set has to be reproducible, and a local Ollama's contents are neither
    # reproducible nor anybody else's business.
    from transcriber_studio import ai_providers

    ai_providers.list_models = lambda settings, provider: {
        "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
        "openai": ["gpt-5", "gpt-5-mini"],
        "ollama_local": ["llama3.3:70b", "qwen2.5:14b"],
    }.get(provider, ["model-a", "model-b"])
    ai_providers.configured_providers = lambda settings: ["anthropic", "openai"]


def main() -> None:
    print("writing screenshots:")
    stub_live_calls()
    # A sandbox that has already been through setup, so nothing prompts.
    config.save(config.Settings(setup_complete=True))
    settings = demo_settings()
    acme, vendor = demo_glossaries()
    settings.glossary_shared_id = acme.id

    window = MainWindow()
    # The options panel is built from settings during construction, so the demo
    # settings have to be pushed in and the panel rebuilt, or the marquee
    # screenshot shows the pipeline switched off.
    window.settings = settings
    window._after_settings_changed()
    window.recordings_tab._on_loaded(demo_recordings())
    demo_jobs(window)
    window._update_job_actions()
    shot(window, "01-main-window", 1320, 880)

    # Just the pipeline group: the whole options panel is two thousand pixels
    # tall and nothing below the fold is the point.
    panel = OptionsPanel(settings)
    panel.resize(560, 2100)
    panel.show()
    for _ in range(6):
        app.processEvents()
    pipeline = next(
        box for box in panel.findChildren(QGroupBox) if box.title() == "Audio pipeline"
    )
    shot(pipeline, "02-audio-pipeline")
    panel.hide()

    library = GlossaryLibraryDialog(selected=acme.id)
    library._merge_into(glossary_store.load(acme.id), [Part.of(acme), Part.of(vendor)])
    shot(library, "03-glossary-review", 1000, 640)

    components = InstallHelpDialog(settings=settings)
    # The dialog runs a local scan first and queues the network check behind
    # it, so waiting on the first worker is not waiting for the versions.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        app.processEvents()
        if any(s.latest for s in components._statuses):
            break
        time.sleep(0.2)
    for _ in range(10):
        app.processEvents()
    shot(components, "04-components-updates", 860, 900)

    cleanup_settings = demo_settings()
    cleanup_settings.glossary_shared_id = acme.id
    cleanup = AICleanupDialog(cleanup_settings, has_original=True, has_cleaned=False)
    if cleanup._model_worker is not None:
        cleanup._model_worker.wait(20000)
    for _ in range(10):
        app.processEvents()
    shot(cleanup, "05-ai-cleanup", 560, 300)

    dialog = SettingsDialog(demo_settings())
    shot(dialog, "06-settings", 780, 900)

    shutil.rmtree(SANDBOX, ignore_errors=True)
    print(f"done — sandbox {SANDBOX} removed")


if __name__ == "__main__":
    main()
