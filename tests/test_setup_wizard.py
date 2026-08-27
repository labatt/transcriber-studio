# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The wizard must collect answers accurately and never write on the way out."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # tests run without a display

from PySide6.QtWidgets import QApplication

from transcriber_studio import whisper_models
from transcriber_studio.config import Settings
from transcriber_studio.ui.setup_wizard import SetupWizard

_app = QApplication.instance() or QApplication([])


def wizard(**settings_kw) -> tuple[SetupWizard, Settings]:
    s = Settings(**settings_kw)
    return SetupWizard(s), s


def test_both_keys_are_asked_for_whichever_engine_is_picked():
    """Either service may be set up now and chosen between per run later."""
    for engine in ("local", "elevenlabs"):
        w, _ = wizard()
        page = w.page(SetupWizard.PAGE_ENGINE)
        page.engine.setCurrentIndex(page.engine.findData(engine))
        keys = w.page(SetupWizard.PAGE_KEYS)
        assert keys.elevenlabs is not None and keys.huggingface is not None


def test_neither_key_is_required_to_continue():
    w, _ = wizard()
    page = w.page(SetupWizard.PAGE_ENGINE)
    page.engine.setCurrentIndex(page.engine.findData("elevenlabs"))
    assert page.isComplete(), "an engine without its key is a warning, not a wall"
    assert w.page(SetupWizard.PAGE_KEYS).isComplete()


def test_a_key_can_only_be_tested_once_something_is_typed():
    w, _ = wizard()
    keys = w.page(SetupWizard.PAGE_KEYS)
    assert not keys.elevenlabs.test_btn.isEnabled()
    assert not keys.huggingface.test_btn.isEnabled()
    keys.huggingface.set_value("hf_x")
    assert keys.huggingface.test_btn.isEnabled()
    assert not keys.elevenlabs.test_btn.isEnabled(), "an empty box is not something to check"
    keys.huggingface.set_value("")
    assert not keys.huggingface.test_btn.isEnabled()


def test_testing_the_entered_keys_skips_the_empty_ones():
    w, _ = wizard()
    keys = w.page(SetupWizard.PAGE_KEYS)
    keys._test_entered()
    assert "Nothing to test" in keys.summary.text()
    # Swap in testers that record instead of hitting the network.
    called = []
    keys.elevenlabs.tester = lambda key: called.append("el") or "ok"
    keys.huggingface.tester = lambda key: called.append("hf") or "ok"
    keys.huggingface.set_value("hf_x")
    keys._test_entered()
    assert "HuggingFace" in keys.summary.text()
    assert "ElevenLabs" not in keys.summary.text()


def test_both_keys_are_saved():
    w, s = wizard()
    keys = w.page(SetupWizard.PAGE_KEYS)
    keys.elevenlabs.set_value("sk_one")
    keys.huggingface.set_value("hf_two")
    w.accept()
    assert s.elevenlabs_api_key == "sk_one"
    assert s.hf_token == "hf_two"


def test_the_model_list_marks_what_suits_this_machine():
    w, _ = wizard()
    page = w.page(SetupWizard.PAGE_ENGINE)
    labels = [page.model.itemText(i) for i in range(page.model.count())]
    assert sum("recommended" in text for text in labels) == 1
    # every entry still carries the bare id as its data
    assert [page.model.itemData(i) for i in range(page.model.count())] == whisper_models.ORDER


def test_picking_a_model_explains_what_it_costs():
    w, _ = wizard()
    page = w.page(SetupWizard.PAGE_ENGINE)
    page.model.setCurrentIndex(page.model.findData("tiny"))
    tiny = page.model_note.text()
    page.model.setCurrentIndex(page.model.findData("large-v3"))
    large = page.model_note.text()
    assert tiny != large
    assert "39M" in tiny and "1550M" in large
    assert "VRAM" in tiny


def test_finishing_writes_every_page_into_settings():
    w, s = wizard()
    engine = w.page(SetupWizard.PAGE_ENGINE)
    engine.engine.setCurrentIndex(engine.engine.findData("elevenlabs"))
    engine.el_model.setCurrentText("scribe_v2")
    w.page(SetupWizard.PAGE_KEYS).elevenlabs.set_value("sk_key")
    speakers = w.page(SetupWizard.PAGE_SPEAKERS)
    speakers.enabled.setChecked(True)
    speakers.maxspk.setValue(5)
    cleanup = w.page(SetupWizard.PAGE_CLEANUP)
    cleanup.enabled.setChecked(True)
    cleanup.provider.setCurrentIndex(cleanup.provider.findData("openai"))
    cleanup.key.setText("sk-openai")
    output = w.page(SetupWizard.PAGE_OUTPUT)
    output.template.setText("{name}")

    w.accept()

    assert s.stt_engine == "elevenlabs"
    assert s.elevenlabs_api_key == "sk_key"
    assert s.elevenlabs_model == "scribe_v2"
    assert s.max_speakers == 5
    assert s.ai_cleanup_enabled and s.ai_cleanup_provider == "openai"
    assert s.ai_key_openai == "sk-openai"
    assert s.filename_template == "{name}"
    assert s.setup_complete is True


def test_cancelling_leaves_the_saved_settings_alone():
    w, s = wizard(output_dir=r"C:\keep", elevenlabs_api_key="sk_original")
    before = s.to_dict()
    w.page(SetupWizard.PAGE_OUTPUT).out_dir.setText(r"C:\discard")
    w.page(SetupWizard.PAGE_KEYS).elevenlabs.set_value("sk_discard")
    w.reject()
    assert s.to_dict() == before


def test_the_key_field_follows_the_chosen_cleanup_provider():
    """Each provider has its own key; switching must not carry one across."""
    w, s = wizard(ai_key_anthropic="sk-ant", ai_key_openai="sk-oai")
    page = w.page(SetupWizard.PAGE_CLEANUP)
    page.provider.setCurrentIndex(page.provider.findData("anthropic"))
    assert page.key.text() == "sk-ant"
    page.provider.setCurrentIndex(page.provider.findData("openai"))
    assert page.key.text() == "sk-oai"


def test_ollama_local_takes_a_url_instead_of_a_key():
    w, s = wizard()
    page = w.page(SetupWizard.PAGE_CLEANUP)
    page.provider.setCurrentIndex(page.provider.findData("ollama_local"))
    assert page.key.text() == s.ollama_local_url
    page.key.setText("http://box:11434")
    w.accept()
    assert s.ollama_local_url == "http://box:11434"


def test_the_summary_names_what_is_still_missing():
    w, _ = wizard()
    engine = w.page(SetupWizard.PAGE_ENGINE)
    engine.engine.setCurrentIndex(engine.engine.findData("elevenlabs"))
    finish = w.page(SetupWizard.PAGE_FINISH)
    finish.initializePage()
    assert "No ElevenLabs API key" in finish.summary.text()


def test_the_summary_reads_the_draft_not_the_saved_settings():
    w, s = wizard(output_dir=r"C:\old")
    w.page(SetupWizard.PAGE_OUTPUT).out_dir.setText(r"C:\new")
    finish = w.page(SetupWizard.PAGE_FINISH)
    finish.initializePage()
    assert r"C:\new" in finish.summary.text()
    assert s.output_dir == r"C:\old", "nothing is saved before Finish"
