# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""An interrupted run resumes from the last completed send."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tests.support import isolated_resume_dir
from transcriber_studio import ai_cleanup
from transcriber_studio import resume as resume_store
from transcriber_studio.config import Settings
from transcriber_studio.jobs import JobRunner
from transcriber_studio.models import Recording, Segment, Source, TranscriptResult
from transcriber_studio.resume import TRANSCRIPT_STAGE, ResumeLog
from transcriber_studio.transcriber import TranscribeOptions

BATCH = [
    Segment(0.0, 2.0, "we pushed the release", "Speaker 1"),
    Segment(2.0, 4.0, "tuesday", "Speaker 1"),
]
RESPONSE = json.dumps({
    "segments": [
        {"from_index": 0, "to_index": 1, "speaker": "Chris",
         "text": "We pushed the release Tuesday."}
    ]
})
PREFIX = "SPEAKER ROSTER:\nSpeaker 1 -> Chris\n\n"


def _recording() -> Recording:
    return Recording(source=Source.LOCAL, id="call.wav", name="Call", date="2026-08-19",
                     local_path="call.wav")


def _log(tmp: str) -> ResumeLog:
    return ResumeLog(Path(tmp) / "r.jsonl")


def test_entries_survive_a_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        log = _log(tmp)
        log.record("k1", "first", stage="cleanup")
        log.record("k2", "second", stage="cleanup")
        reopened = ResumeLog(Path(tmp) / "r.jsonl").load()
        assert reopened.get("k1") == "first"
        assert reopened.get("k2") == "second"
        assert reopened.count("cleanup") == 2


def test_a_torn_final_line_is_ignored_not_fatal():
    """A hard kill mid-write truncates the last line; earlier sends still count."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        log = ResumeLog(path)
        log.record("k1", "complete", stage="cleanup")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"key": "k2", "stage": "clea')     # power cut mid-line
        reopened = ResumeLog(path).load()
        assert reopened.get("k1") == "complete"
        assert reopened.get("k2") is None


def test_restored_batch_makes_no_api_call():
    with tempfile.TemporaryDirectory() as tmp:
        log = _log(tmp)
        key = ai_cleanup._cleanup_send_key("anthropic", "claude-opus-5", PREFIX, BATCH)
        log.record(key, RESPONSE, stage="cleanup")

        # provider "exploding" has no credentials and no client; reaching the
        # network at all would raise rather than quietly cost tokens.
        out = ai_cleanup._cleanup_chunk_once(
            BATCH, ["Speaker 1"], Settings(), "anthropic", "claude-opus-5",
            None, PREFIX, None, resume=log,
        )
        assert [s.text for s in out] == ["We pushed the release Tuesday."]
        assert [s.speaker for s in out] == ["Chris"]


def test_a_changed_batch_is_not_restored_from_a_stale_answer():
    key_a = ai_cleanup._cleanup_send_key("anthropic", "opus", PREFIX, BATCH)
    changed = [Segment(0.0, 2.0, "we shipped the release", "Speaker 1"), BATCH[1]]
    key_b = ai_cleanup._cleanup_send_key("anthropic", "opus", PREFIX, changed)
    assert key_a != key_b


def test_a_changed_glossary_invalidates_saved_batches():
    key_a = ai_cleanup._cleanup_send_key("anthropic", "opus", PREFIX, BATCH)
    key_b = ai_cleanup._cleanup_send_key(
        "anthropic", "opus", "SPEAKER ROSTER:\nSpeaker 1 -> Mark\n\n", BATCH
    )
    assert key_a != key_b


def test_a_different_model_invalidates_saved_batches():
    key_a = ai_cleanup._cleanup_send_key("anthropic", "claude-opus-5", PREFIX, BATCH)
    key_b = ai_cleanup._cleanup_send_key("anthropic", "claude-sonnet-5", PREFIX, BATCH)
    assert key_a != key_b


def test_manual_cancel_drops_sends_but_keeps_the_transcript():
    with tempfile.TemporaryDirectory() as tmp:
        log = _log(tmp)
        log.record("t", "{}", stage=TRANSCRIPT_STAGE)
        log.record("c1", "x", stage="cleanup")
        log.record("c2", "y", stage="cleanup")
        dropped = log.discard(keep_stages=(TRANSCRIPT_STAGE,))
        assert dropped == 2
        reopened = ResumeLog(Path(tmp) / "r.jsonl").load()
        assert reopened.get("t") == "{}"
        assert reopened.get("c1") is None
        assert reopened.count("cleanup") == 0


def test_full_discard_removes_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        log = _log(tmp)
        log.record("c1", "x", stage="cleanup")
        log.discard()
        assert not (Path(tmp) / "r.jsonl").exists()


def test_transcript_is_restored_instead_of_re_running_whisper():
    class _Exploding:
        def transcribe(self, *a, **k):
            raise AssertionError("Whisper must not run when a transcript is saved")

    with isolated_resume_dir(), tempfile.TemporaryDirectory() as tmp:
        rec = _recording()
        runner = JobRunner.__new__(JobRunner)
        runner.s = Settings()
        runner.transcriber = _Exploding()
        runner.client = None

        log = _log(tmp)
        key = resume_store.transcript_key(rec, runner._opts())
        saved = TranscriptResult(
            recording=rec,
            segments=[Segment(0.0, 2.0, "Hello.", "Speaker 1")],
            language="en",
            model="large-v3",
        )
        log.record(
            key,
            json.dumps(resume_store.transcript_to_dict(saved)),
            stage=TRANSCRIPT_STAGE,
        )

        out = runner._transcribe_or_restore(rec, 1, None, None, None, log)
        assert [s.text for s in out.segments] == ["Hello."]
        assert out.language == "en"


def test_transcript_key_changes_with_the_options_that_change_output():
    rec = _recording()
    base = TranscribeOptions(model="large-v3", language="auto", diarization_enabled=True)
    other_model = TranscribeOptions(model="medium", language="auto", diarization_enabled=True)
    no_diar = TranscribeOptions(model="large-v3", language="auto", diarization_enabled=False)
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, other_model)
    assert resume_store.transcript_key(rec, base) != resume_store.transcript_key(rec, no_diar)
    assert resume_store.transcript_key(rec, base) == resume_store.transcript_key(rec, base)


def _cleanup_response(segments, offset=0):
    return json.dumps({
        "segments": [
            {"from_index": i, "to_index": i, "speaker": "Chris",
             "text": f"Clean {offset + i}."}
            for i in range(len(segments))
        ]
    })


def test_interrupted_cleanup_resends_only_the_missing_batches():
    """The whole point: a run that dies mid-cleanup costs nothing to finish."""
    import transcriber_studio.ai_providers as ai_providers
    from transcriber_studio.ai_store import ModelProfile

    rec = _recording()
    transcript = TranscriptResult(
        recording=rec,
        segments=[Segment(float(i), float(i + 1), f"raw {i}", "Speaker 1") for i in range(6)],
        language="en",
        model="large-v3",
    )

    settings = Settings()
    settings.glossary_enabled = False          # isolate the cleanup stage
    settings.ai_cleanup_provider = "openai"
    settings.ai_cleanup_model = "gpt-4o"
    settings.prompt_cache_enabled = False

    sent: list[str] = []
    die_after = {"n": 2}

    def fake_completion(settings_, provider, model, system, user, profile, **kw):
        if len(sent) >= die_after["n"]:
            raise ConnectionError("Connection aborted — the machine went down")
        sent.append(user)
        return _cleanup_response(json.loads(user)["segments"], offset=len(sent) * 10)

    originals = (
        ai_providers.chat_completion,
        ai_providers.is_provider_configured,
        ai_providers.list_models,
        ai_cleanup.load_profile,
        ai_cleanup.save_profile,
        ai_cleanup.RELIABLE_CLEANUP_BATCH_SEGMENTS,
    )
    try:
        ai_providers.chat_completion = fake_completion
        ai_providers.is_provider_configured = lambda s, p: True
        ai_providers.list_models = lambda s, p: []
        ai_cleanup.load_profile = lambda p, m: ModelProfile(provider=p, model_id=m)
        ai_cleanup.save_profile = lambda p: None
        ai_cleanup.RELIABLE_CLEANUP_BATCH_SEGMENTS = 2     # 6 segments -> 3 batches

        with isolated_resume_dir():
            # --- run 1: dies after two batches ---
            try:
                ai_cleanup.cleanup_transcript(transcript, settings)
                raise AssertionError("expected the simulated crash")
            except AssertionError:
                raise
            except Exception:
                pass
            assert len(sent) == 2, "two batches should have completed before the crash"

            saved = resume_store.log_for(rec)
            assert saved.count("cleanup") == 2, "both completed batches must be on disk"

            # --- run 2: the crash is over, finish the job ---
            die_after["n"] = 99
            sent.clear()
            fresh = TranscriptResult(
                recording=rec,
                segments=[Segment(float(i), float(i + 1), f"raw {i}", "Speaker 1")
                          for i in range(6)],
                language="en",
                model="large-v3",
            )
            out = ai_cleanup.cleanup_transcript(fresh, settings)

            assert len(sent) == 1, f"only the unfinished batch should be re-sent, got {len(sent)}"
            assert len(out.segments) == 6
            assert resume_store.resume_path(rec).exists() is False, "finished run clears up"
    finally:
        (ai_providers.chat_completion, ai_providers.is_provider_configured,
         ai_providers.list_models, ai_cleanup.load_profile, ai_cleanup.save_profile,
         ai_cleanup.RELIABLE_CLEANUP_BATCH_SEGMENTS) = originals


def test_disabled_log_is_a_no_op():
    log = ResumeLog(None)
    log.record("k", "v", stage="cleanup")
    assert log.get("k") == "v"      # in-memory only
    assert log.discard() == 1


def test_progress_summary_reports_each_stage():
    with isolated_resume_dir():
        rec = _recording()
        log = resume_store.log_for(rec)
        assert resume_store.describe_progress(rec) == ""
        log.record("t", "{}", stage=TRANSCRIPT_STAGE)
        log.record("g", "{}", stage="glossary")
        log.record("c1", "{}", stage="cleanup")
        log.record("c2", "{}", stage="cleanup")
        assert resume_store.saved_progress(rec) == {
            "transcript": 1, "glossary": 1, "cleanup": 2,
        }
        assert resume_store.describe_progress(rec) == (
            "transcript, 1 glossary chunk(s), 2 cleanup batch(es)"
        )


def test_progress_summary_refreshes_when_the_log_grows():
    """The mtime cache must not hide progress made after the first read."""
    with isolated_resume_dir():
        rec = _recording()
        log = resume_store.log_for(rec)
        log.record("c1", "{}", stage="cleanup")
        assert resume_store.saved_progress(rec) == {"cleanup": 1}
        import os
        import time as _t
        log.record("c2", "{}", stage="cleanup")
        os.utime(resume_store.resume_path(rec), (_t.time() + 1, _t.time() + 1))
        assert resume_store.saved_progress(rec) == {"cleanup": 2}


def test_progress_summary_is_empty_once_the_job_finishes():
    with isolated_resume_dir():
        rec = _recording()
        log = resume_store.log_for(rec)
        log.record("c1", "{}", stage="cleanup")
        assert resume_store.describe_progress(rec) != ""
        log.discard()
        assert resume_store.describe_progress(rec) == ""


def test_batches_saved_after_a_split_are_reused_by_the_next_run():
    """The real-world failure: a run split 600 -> 150 and saved the 150s.

    A fresh run must not start at 600 and re-send everything; it has to line
    up with the sizes already on disk.
    """
    import transcriber_studio.ai_providers as ai_providers
    from transcriber_studio import ai_cleanup as ac
    from transcriber_studio.ai_store import ModelProfile

    rec = _recording()
    segs = [Segment(float(i), float(i + 1), f"raw {i}", "Speaker 1") for i in range(24)]
    transcript = TranscriptResult(recording=rec, segments=segs, language="en", model="w")

    settings = Settings()
    settings.glossary_enabled = False
    settings.ai_cleanup_provider = "openai"
    settings.ai_cleanup_model = "gpt-4o"
    settings.prompt_cache_enabled = False

    sent_sizes: list[int] = []

    def fake_completion(settings_, provider, model, system, user, profile, **kw):
        payload = json.loads(user)["segments"]
        sent_sizes.append(len(payload))
        return _cleanup_response(payload)

    originals = (
        ai_providers.chat_completion, ai_providers.is_provider_configured,
        ai_providers.list_models, ac.load_profile, ac.save_profile,
        ac.RELIABLE_CLEANUP_BATCH_SEGMENTS,
    )
    try:
        ai_providers.chat_completion = fake_completion
        ai_providers.is_provider_configured = lambda s, p: True
        ai_providers.list_models = lambda s, p: []
        ac.load_profile = lambda p, m: ModelProfile(provider=p, model_id=m)
        ac.save_profile = lambda p: None
        ac.RELIABLE_CLEANUP_BATCH_SEGMENTS = 12     # a fresh run would use 12s

        with isolated_resume_dir():
            # Seed the log as if an earlier run split 12 -> 6 and saved two 6s.
            log = resume_store.log_for(rec)
            prefix = ac.CLEANUP_USER_INSTRUCTIONS
            for chunk in (segs[0:6], segs[6:12]):
                key = ac._cleanup_send_key("openai", "gpt-4o", prefix, chunk)
                log.record(key, _cleanup_response(
                    [{"index": i} for i in range(len(chunk))]
                ), stage="cleanup", segments=len(chunk))

            out = ac.cleanup_transcript(transcript, settings)

            assert 6 not in [] and all(n <= 6 for n in sent_sizes), (
                f"should match the saved 6-segment size, sent {sent_sizes}"
            )
            # 24 segments, 6 per batch = 4 batches; 2 were already saved.
            assert len(sent_sizes) == 2, f"only 2 batches should be sent, sent {len(sent_sizes)}"
            assert len(out.segments) == 24
    finally:
        (ai_providers.chat_completion, ai_providers.is_provider_configured,
         ai_providers.list_models, ac.load_profile, ac.save_profile,
         ac.RELIABLE_CLEANUP_BATCH_SEGMENTS) = originals
