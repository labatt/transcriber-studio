# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Gemini 3.5 Transcribe** as a third transcription engine, alongside local Whisper and ElevenLabs
  Scribe. It transcribes and separates speakers in one pass and uses the same Google AI key as AI
  Cleanup. Verbatim mode is the default because it is the only one Google lets return speakers and
  timestamps; smart mode returns prose and the app says so rather than producing empty timings.
- **An installer.** `install.ps1` (Windows) and `install.sh` (macOS/Linux) find or install a
  suitable Python and hand over to `install.py`, which detects the OS, package manager, GPU and
  driver, then checks each requirement's version and installs or upgrades only what is missing.
  `--check`, `--dry-run`, `--yes`, `--minimal` and `--no-gpu` are all supported. Where there is no
  package manager it downloads what it needs directly, including a static ffmpeg build on Windows
  and the DeepFilterNet binary for the running platform.

### Fixed

- **A job could sit forever after a suspend/resume.** DeepFilterNet processes a whole recording in
  one pass and writes nothing until it finishes, so a machine sleeping mid-run left the app waiting
  on a child that had stopped making progress, with no error, no progress and no way out but
  Cancel. Denoising now runs in two-minute chunks with a timeout on each, reports progress as it
  goes, and a chunk that stalls falls back to its own original audio — the recording stays complete
  and correctly timed, just less clean over that stretch.
- **Interrupted work is no longer thrown away.** Finished denoise chunks are kept between runs, so
  restarting an interrupted job resumes instead of starting over. Speaker turns are cached the same
  way. Downloads keep their partial file and resume with an HTTP range request rather than
  re-fetching an hour-long recording from the beginning.
- **A crash during speaker detection no longer costs you the transcription.** The Whisper pass is
  checkpointed before diarization starts rather than after it finishes, so the minutes of GPU time
  that produced the words survive a failure in the stage that only labels them. Resume offers
  "transcribed audio (speakers still to do)" and picks up from there. The checkpoint deliberately
  ignores the diarization settings, since speaker labels are attached to the segments afterwards.
- **Speakers are assigned per word instead of per segment.** Whisper cuts segments on pauses and
  punctuation, never on speaker changes, so a segment routinely holds two people. Labelling the
  whole segment by whichever speaker overlapped it most filed one person's words under the other's
  name, which on a fast back-and-forth is most of the transcript. Word timings were already being
  computed and then discarded; they are now kept, each word is matched to a diarization turn, and
  the segment is split where the floor changes hands. Falls back to the old behaviour when word
  timestamps are off. Resume checkpoints carry the word timings so a resumed job labels speakers
  as well as one that ran straight through.
- **Speaker detection can be cancelled.** pyannote runs as one long call; the app now interrupts it
  through the progress hook, and the Cancel button is wired to it. A cancel is also no longer
  swallowed by the handler that skips past a failed diarization.
- **The progress bar jumped backwards when transcription started.** Fetching and cleaning the audio
  already own the first 40% of the bar, but the decoder reported its own 0-to-1 over the top of
  them, so the bar fell to zero and crawled for the longest stage of the job — which reads as
  stuck. Decoding now maps to the 40-92% it actually owns.
- **The decoder says how far it has got.** An hour of audio is many minutes with nothing in the log
  between "Transcribing audio…" and the next stage. It now reports minutes done, percentage and
  segment count every 30 seconds, so the log is evidence of work rather than a gap.
- Every ffmpeg conversion has a time limit, so no stage of the pipeline can wait indefinitely.

- **Gemini transcribes recordings of any length**, by cutting them into 30-minute parts and joining
  the results. Google documents a 30-minute ceiling and does not enforce it: probing
  gemini-3.5-transcribe in verbatim mode with diarization, 54 minutes was accepted and 57 was
  refused with a bare "Invalid input received." — the same refusal for a 152 MB wav and a 38 MB mp3
  of the same audio, so it is duration, not size. Parts are cut at a pause near the boundary rather
  than on the clock, so a word is not sliced in half at every seam, and each part's timings are
  shifted back onto the original timeline before the whole transcript is grouped into turns.
  Each part repeats the last three minutes of the one before it. Gemini numbers speakers within a
  single request and offers no way to tell it about the others — there is no enrollment API to pass
  reference voices to — but the same speech described twice by two requests is enough to work out
  which of their speaker numbers mean the same person. The repeated words are then discarded, so the
  transcript still breaks exactly on the silence-aligned seam. Diarization stays Gemini's throughout;
  pyannote plays no part on this path. A join where only one person speaks can only match that one,
  and the log says how many voices it could not place rather than implying they were matched.

- **The Options column now holds only what you choose per job**, and the Settings dialog is tabbed.
  The column had seven group boxes and around forty-five controls in a scrolling strip, mixing
  per-run choices with tuning set once and forgotten. It now carries the engine, the three pipeline
  switches, speaker labels, formats, output folder and AI cleanup — with an "Advanced settings…"
  button for the rest. VAD tuning, vocabulary terms, channels, line formatting, file naming and
  glossary tuning moved into Settings, which gained tabs (Engines, Audio, Speakers, Output,
  AI Cleanup, Plaud) rather than becoming the new long scroll.

- **Speaker count moved into the Options column**, where the per-job choices are. How many people
  are in a recording is a fact about that recording, not a preference — and telling the diarizer is
  the largest single influence on how well it labels them. It reads "How many: at least / at most",
  both defaulting to auto, and Settings no longer carries a second copy.

### Changed

- The manual install instructions cover Windows, macOS and Linux rather than assuming `winget`.
- `transcriber_studio.components` now reaches the network through the standard library when
  `requests` is absent, so the installer and the component registry share one implementation
  instead of the installer duplicating it.

## [0.1.0] — 2026-08-27

First public release.

### Added

- **Audio pipeline in front of the decoder**: DeepFilterNet denoising (standalone binary, pip
  package, or an ffmpeg fallback), Silero VAD via faster-whisper, and vocabulary biasing that feeds
  glossary terms to the decoder as hotwords. Plus a hallucination guard that stops one invented
  passage seeding the next window.
- **Shared glossaries**: named vocabularies that jobs read from and write back to, with import,
  combine, dedupe, and a review queue for entries two sources disagree about.
- **CrisperWhisper** as a model option — a verbatim `large-v3` fine-tune (English/German,
  non-commercial weights).
- **AI cleanup** via OpenRouter, OpenAI, Anthropic, Google, xAI or Ollama, with an app-wide default
  provider and model.
- **Components window**: installed versions against current releases, with install/update buttons,
  a CUDA-aware PyTorch upgrade, elevation handling, and PATH refresh from the registry.
- **Resume**: interrupted jobs restart from their last saved step.
- Output as `txt`, `srt`, `vtt`, `json` or `md`, with a filename template builder.

### Changed

- Variant dedupe in glossary merges no longer drops spellings that collapse to the canonical form —
  `growth mark` under `GrowthMark` is exactly the misspelling the cleanup model needs. Selection is
  order-based so the same inputs always produce the same output.
- ffmpeg and other CLI tools are resolved when used rather than at import, so an upgrade that moves
  the install directory does not make the app report a missing dependency.

### Notes

Renamed from an earlier private build; an existing `%APPDATA%\PlaudWhisperStudio` directory is
copied to `%APPDATA%\TranscriberStudio` on first run.
