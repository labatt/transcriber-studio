# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Identify speakers by ear before transcribing.** *Go with options…* next to Go opens a
  dialog for this run only: how many people were in the room, whether to denoise, detect
  speakers or run AI cleanup, and *Identify speakers first*. With that ticked, each recording's
  speakers are detected before a word is transcribed and a dialog plays two to four short
  samples of each voice, taken from far-apart parts of the recording, next to a name field.
  The names go on the transcript — a name heard beats a name measured — and ticked voices are
  stored as voiceprints, so the next recording names those people on its own. Speakers already
  recognised from an enrolled voice arrive pre-filled. Detection runs once: the transcription
  that follows reuses it from the cache. *…and stop there* does only the identification —
  no engine, no files — for teaching the app a room full of voices; names given that way are
  kept for the session and used by a later Go on the same recording. *Only the first N
  minutes* (ten by default) cuts every recording before any stage sees it — a cheap way to
  try an engine, a glossary or the denoiser on a long meeting; the cut is a stream copy and
  is never mistaken for the full transcript when resuming.
- **ElevenLabs Scribe with pyannote speakers.** Scribe now works the way MAI does: it supplies
  the words and pyannote separates the speakers locally, so voiceprints work on ElevenLabs
  transcripts too. Scribe's own speaker labels are still available from Settings → Engines, and
  are used automatically when no HuggingFace token is saved. The default Scribe model is now
  `scribe_v2`; v1 is deprecated upstream.
- **Download audio** button: save the ticked PLAUD recordings' audio to a folder of your choice
  without transcribing. The file also lands in the audio cache, so transcribing it later costs no
  second download.

### Fixed

- **PLAUD recordings that are really Ogg/Opus.** PLAUD's cloud now serves Opus-in-Ogg under a
  name ending in `.mp3`. Everything that chose a container or a MIME type from the filename —
  the new first-N-minutes cut, Gemini's long-file splitting, Gemini's upload type — now reads
  the format from the bytes instead. The cut also falls back to a lossless WAV when a stream
  copy is impossible, and a zero-byte leftover from a failed cut is never reused.

- **MAI-Transcribe-2** (Microsoft, via Azure Speech) as a fourth transcription engine. It takes a
  whole recording in one request (a 73-minute, 70 MB file went up unsplit), accepts up to 50
  vocabulary hints from the glossary, and was the most accurate engine on every recording in the
  benchmark below. Its own speaker diarization fails on anything much past 15 minutes with errors
  that read like network timeouts, so by default the app uses MAI for the words and pyannote for
  the speakers, which also makes voiceprints work on it. Reported to Microsoft
  (`docs/mai-transcribe-diarization-issue.md`, MicrosoftDocs/azure-ai-docs#835). Picking MAI turns
  denoising off for that run and says so.
- **Voiceprints.** pyannote's per-speaker voice embeddings are kept on every transcript. Name a
  speaker in the rename dialog, tick *Remember this voice*, and later recordings come back with
  the name instead of `Speaker 2`. A **Speakers** button in the header opens a dialog for the
  enrolled voices, the match thresholds, and a log of every match decision.
- **Detect speakers again.** *Detect speakers* now works on a job that already has speakers, and
  asks for the *at least / at most* limits for that run only, so a six-person meeting transcribed
  as two because the previous call's limit was still set can be redone without re-transcribing.
  Transcripts keep their word timings so the turns are regrouped word by word; a transcript from
  before words were kept gets them back by force-aligning its text to the audio (torchaudio's
  wav2vec2 aligner, English; median start error 59 ms against AMI's hand-timed words), with even
  spacing as the fallback.
- **Rename PLAUD recordings** from the list and, opt-in, push the new name to the PLAUD account
  through its web interface (the public API is read-only). **Rename output files** from the Output
  column.
- **Glossary weighting.** A term seen again gains weight, and the heaviest terms are the ones
  biased first when the hint list has to be cut to fit the decoder.
- **Repetition controls and confidence data** for Whisper: a repetition penalty and a no-repeat
  n-gram size in Settings, and per-segment confidence kept on the transcript.
- **An engine benchmark** (`tools/engine_bench.py`, `tools/bench_refs.py`): every engine over
  recordings with reference transcripts from the AMI Meeting Corpus and Earnings-22, scored on word
  error rate, proper-noun recall, speaker attribution and diarization error rate. Results, an
  issues log, a process log and the write-up are under `docs/`.
- **`large-v3-turbo`** in the model list. In the benchmark it scored within two points of
  `large-v3` on every recording.

### Changed

- **Gemini requests are capped at the documented 30 minutes**, lead-in included. The API accepts
  up to about 54 minutes with speakers and word timings, but past 30 it returned word timings out
  of order, starts hours beyond the end of the file, and a 17-word sentence repeated 60 times;
  scored against a reference transcript that was 38 percent word error rate against 8 percent for
  the same call cut at 30. Stray timings and decoding loops that remain are repaired and logged.
- **AI Cleanup batches are sized to the answer's real length.** They were budgeted as if the
  answer were a quarter of the text; it is the whole text, so 95k-character batches were cut off
  at the output limit and retried four times before splitting. Answers are budgeted in full, a
  cut-off answer splits at once, the split log line says why, and Gemini and OpenAI's GPT-5.6
  family get 32,768-token answers (their documented limits are 65,536 and 128,000).
- **OpenAI's reasoning models are known in advance.** GPT-5.6 Sol, Terra and Luna and GPT-6 get
  `max_completion_tokens`, no temperature and a low reasoning effort from the first request,
  instead of learning each by a rejected send; a quirk learned on one batch now carries to the
  next batch in the same run.
- **Uploads show progress and get a timeout sized to the file.** urllib3 applies the connect
  timeout to writing the request body, so a 70 MB upload died with "The write operation timed
  out" at a fixed timeout.
- **Speaker rosters stay with their recording.** A shared glossary no longer lends rows keyed by
  a diarization label (`Speaker 2 = Greg`) to other recordings, only rows with a stable label and
  a name; and a recording's saved roster is re-extracted, terms kept, when its labels no longer
  match the transcript because speakers were re-detected. Before this, one person turned up as a
  speaker in meetings they were never in, and a re-detected seven-speaker transcript came out
  with fourteen labels.
- **One spelling per person after cleanup.** The model can return the same speaker under two
  spellings; case and spacing variants are unified, near-identical spellings are folded toward
  the roster's or the commonest, generic labels and distinct roster names are never merged, and
  each merge is logged. Renaming two labels to one person yields one speaker.
- **Banked transcripts are keyed by what the engine reads.** A Scribe job quit during cleanup
  re-uploaded the whole hour on Resume because the key included the vocabulary hint text, which
  the job's own glossary stage had changed. Keys now hold only the settings the engine in use
  reads, with hints reduced to on/off, and a transcript banked under an older key formula is
  still found by its engine and model.
- **Interrupted jobs are reconciled at startup.** A job that was running when the app closed is
  marked *Interrupted* rather than left showing *In progress* forever.
- The main window can be resized below its old minimum width; the header and job buttons wrap.

### Fixed

- **Gemini Flash spent its output budget thinking** on cleanup batches (15,728 of 16,384 tokens
  on one real batch) and truncated the JSON. Thinking is turned off for cleanup, with a fallback
  when a model rejects the setting.
- **MAI's phrase list is capped at 50**, undocumented; a 64-term glossary failed every request
  with a 400. The app sends the 50 heaviest terms and logs how many were left out.
- **Logout crashed the app** when the token had expired: a worker thread was destroyed while
  running. Workers are now tracked and given a grace period at close.
- Voiceprint embeddings were lost when a result was copied, queued or resumed; MAI's spacing
  entries crashed speaker assignment; words in pauses got no speaker (now the nearest turn within
  two seconds).

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
