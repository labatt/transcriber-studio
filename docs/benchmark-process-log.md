# Process log: how the engine comparison was run

Chronological notes on what was done and why, 2026-09-05. Companion to
`benchmark-issues-log.md`, which lists what went wrong. Times are local.

## 1. First attempt: the user's own recordings (16:43 to 17:03)

- Wrote `tools/engine_bench.py`. It runs each engine configuration through
  `Transcriber.transcribe`, the same path a job takes, with the saved settings
  (glossary hotwords, VAD, pyannote speaker separation), so results describe
  the app as used rather than a bare API call.
- Six configurations: MAI verbatim, MAI clean, ElevenLabs Scribe, Gemini
  verbatim, Whisper large-v3-turbo, Whisper large-v3. MAI and Whisper use
  local pyannote for speakers.
- Two recordings from the Downloads folder, 20.6 and 50.7 minutes, both Plaud
  exports (16 kHz mono MP3, about 56 kbps).
- Measured per run: wall time, words, segments, speech coverage, speakers,
  speaker turns, glossary terms found, fillers, numbers, repeated adjacent
  segments, most repeated 5-gram, confidence availability, retries, voice
  embeddings returned, and pairwise word error rate between engines.
- Because there is no reference transcript for these files, pairwise WER was
  labelled agreement, not accuracy, in the report.
- ElevenLabs cost control: Scribe bills about 330 tokens per minute against a
  60,000-token month, so a `--cap` option skipped it on the 50-minute file.
- Cloud order first, so a crash during a long local decode would not cost a
  re-upload. Results saved after every run so the script can resume.
- Stopped on request after MAI, Scribe and Gemini had finished; the user asked
  for accuracy against known text instead.

## 2. Choosing a ground truth (17:03 to 17:20)

- Considered a synthetic test: a scripted meeting rendered with text-to-speech,
  seeded with the user's own glossary names, mixed with babble noise and
  reverb at set signal-to-noise ratios. Confirmed the existing Azure Speech key
  serves 196 English neural voices, and `edge-tts` is installed as a fallback.
  Deferred: TTS speech is cleaner than real speech, and a mispronounced name
  fails every engine for the wrong reason.
- Surveyed public corpora with transcripts. Chose AMI (CC BY 4.0, real meetings,
  distant and headset mics, word-level transcripts with speakers) for the
  meeting case and Earnings-22 (Rev, CC BY-SA 4.0 for the transcripts, earnings
  calls dense with names and numbers, many accents) for the vocabulary case.
  Skipped DiPCo (single 13.4 GB archive), CHiME-6 (manual data request) and
  VoxConverse (audio via YouTube, licence unstated).
- Verified licences and download paths before recommending. The Earnings-22
  licence file was not at the repository root; it is in `earnings22/LICENSE.md`.

## 3. Getting the data (17:05 to 17:02 next hour)

- AMI: downloaded two meetings from the Edinburgh mirror, ES2004b (table-array
  channel 1 and headset mix) and EN2002a (table-array), plus the manual
  annotation zip (NXT format). Extracted only `words/`, `segments/` and
  `corpusResources/meetings.xml`.
- Earnings-22: picked two subset10 calls by metadata, one US (4474506, Costco,
  66 min) and one UK (4483937, 61 min). The subset's media entries are symlinks
  to `earnings22/media/`, which is behind Git LFS; the first download produced
  empty files until the LFS media URL used the resolved path. Used the
  force-aligned `.nlp` references (token, speaker, timestamps per token).
- Transcoded the AMI WAVs to 16 kHz mono MP3 at 56 kbps with ffmpeg so the
  engines see the same kind of file the Plaud produces. Earnings-22 audio is
  already low-bitrate MP3 and was left as is.

## 4. Building the scorer (17:05 to 17:02)

- Wrote `tools/bench_refs.py`: parsers that turn AMI NXT and Rev `.nlp` into one
  reference shape (words with text, start, end, speaker; segments with speaker),
  and a `score()` function.
- Metrics: WER after Whisper's English text normaliser (the Open ASR
  Leaderboard method, fetched from the whisper-large-v3 model card), with
  substitutions, deletions and insertions; proper-noun recall (capitalised,
  non-sentence-initial reference words); speaker attribution accuracy on
  aligned words after an optimal speaker mapping (Hungarian assignment);
  diarization error rate via pyannote.metrics with a 0.25 s collar; words
  emitted during reference silence.
- Self-tested on a transcript built from the reference itself. First pass
  exposed two faults: AMI capitalises every utterance start, so common words
  counted as names (fixed by flagging sentence-initial words at parse time);
  and normalising word by word split numbers differently from an engine's
  output (fixed by scoring WER on whole normalised texts and keeping per-token
  alignment only for the metrics that need to know which word matched). After
  the fixes the self-test gives zero WER, 1.0 attribution and zero diarization
  error on the Earnings reference.
- Wired scoring into `engine_bench.py`: a `NAME.ref.json` beside the audio is
  picked up automatically, score rows join the report, and per-engine lists of
  missed proper nouns are printed. Added `--skip NAME=STEM` for per-recording
  exclusions and `--no-hints` so the user's personal glossary is not sent for
  public recordings where it is meaningless.

## 5. Running the cloud engines (17:03 to 17:45)

- Five recordings (the two AMI table-mic files, the two earnings calls, then
  the ES2004b headset mix last, as a microphone-effect control).
- Scribe on the first AMI meeting only, about 12,900 tokens.
- The user asked not to start local runs before a laptop shutdown, so a
  watcher process waited for the first "start whisper" line in the log and
  killed the benchmark there. It fired three seconds into the first Whisper
  run; nothing was lost because results are written after every run.
- Copied results, references and the downloaded audio (81 MB) out of the
  session temp folder to `%APPDATA%\TranscriberStudio\bench\` so a reboot
  could not lose them.

## 6. Investigating the Gemini anomalies (17:45 onward)

- Its diarization error rates were impossible (above 100 percent) and speech
  coverage 17 to 50 times the recording length. Counted segments whose end
  preceded their start (25 across four runs) and found a start of 99,711 s in a
  66-minute file, inside part 1, so not the split offsets.
- Counted repeated eight-word windows per transcript: 3,585 repeated instances
  on the Costco call against MAI's 5, and one phrase 60 times; so the WER gap
  was looping, not mishearing, and the seam neighbourhoods themselves read
  normally.
- Added `tidy_words` to the Gemini module: `repair_timings` and
  `collapse_loops`, applied per part before offsets are added and on the
  single-request path. Unit tests for each fault seen live.
- Re-ran Gemini alone on the two affected recordings into a separate folder so
  the running Whisper benchmark was untouched. First version of the repair
  chained on the previous word and stacked 3,447 words at the one outlier;
  the loop collapse missed because the looped sentence was 17 words and the
  window stopped at 16. Rewrote the repair to judge each start against the
  median of its neighbours, widened the loop window to 40 words and ignored
  punctuation when matching, added tests for both, re-ran again.
- Second re-run: loop collapsed (18-word phrase, 58 repeats), WER 38.0 to 30.1,
  insertions 2,505 to 1,544, diarization error 5,436 percent to 56 percent. But
  2,774 timings still needed repair in part 2, concentrated in the last ten
  minutes of the part (the 50 to 60 minute bucket held 2,738 words in 65
  segments, 54 of them stacked at one instant), so the fault is systematic late
  in a long part, not a stray value.
- Cut part 2 (27 to 57 min) with ffmpeg and sent it to Gemini directly, saving
  the raw response, to characterise the timestamp fault from the source rather
  than from the repaired output. It came back clean: 5,132 words, no timing
  faults, no loop. Sent it twice more to rule out luck: clean both times.
- So the difference had to be in the app's request. Checked the request body
  (no temperature or other generation settings differ) and the part cutter:
  same codec (`-c copy` MP3), same mime type. Ran the cutter on the file and
  measured the parts: 30.0, 33.0 and 9.3 minutes. The middle part was 30
  minutes seam to seam plus the 3-minute lead-in. Every clean request had been
  exactly 30 minutes; every broken one was longer (33, 35.7, 39). The app's
  split threshold was 54 minutes, from an earlier probe of what the API
  accepts, so EN2002a and ES2004b had gone up whole.
- Changed the split threshold to the documented 30 minutes, and made the
  splitter's target the maximum part length including the lead-in (new
  `part_spans` helper; seams 27 minutes apart). Replaced the test that pinned
  the 54-minute figure with one recording the measurements, and added tests
  that no part can exceed the target. Re-ran Gemini on the three affected
  recordings, gated on the full test suite passing (589 passed).
- Result: Costco WER 38.0 to 7.8 percent, attribution 0.51 to 0.98; EN2002a
  43.2 to 33.4; ES2004b unchanged in WER but with a sane speaker timeline.
  Then re-ran Gemini on the remaining two recordings under the same code so
  all five Gemini rows in the final report come from one configuration. The
  original Gemini results are kept in `bench_public` for the before/after.
- Meanwhile, looked at where Whisper turbo loses words on ES2004b: evenly over
  time and over all four speakers, with the VAD reporting 4.7 of 39.1 minutes
  cut as non-speech on a meeting where half the reference utterances are under
  1.5 seconds. Large-v3 then scored the same as turbo on both AMI meetings
  (23.0 and 39.3 against 21.0 and 39.6) with the same word counts, so the
  decoder is not the variable. Added a turbo-without-VAD configuration
  (`--only whisper-large-v3-turbo-novad`, excluded from the default set) and
  ran it on both AMI table-mic meetings once the benchmark had exited: about
  one point of WER and 140 words recovered per file, deletions still far above
  MAI's. Conclusion recorded in the issues log (item 21).

## 8. Assembling the results (20:30 onward)

- With the benchmark exited (its summary file is rewritten after every run,
  so merging while it ran would have been overwritten), folded the fixed
  Gemini runs into the main folder with `merge_gemini.py`, keeping the first
  runs under `gemini-verbatim-unfixed`, and regenerated the report.
- Wrote `docs/engine-benchmark-results.md` (tables, findings, cost, caveats),
  copied the raw report to `docs/engine-benchmark-report.md`, and cross-checked
  every table cell against the summary JSON.

## 9. Completing the Scribe column (21:22 and 22:30)

- Wrote the article (`docs/article-stt-engine-benchmark.md`) and ran the
  humanizer pass over it.
- The user asked for Scribe on EN2002a (second AMI meeting), then decided to
  spend credits on the remaining three recordings. Cleared the "skipped"
  markers from the summary so the resumable run would take them, ran Scribe
  on the headset mix and both earnings calls (about 2 minutes of wall time for
  166 minutes of audio), regenerated the report.
- Scribe came second on all five, 0.1 to 1.2 points behind MAI, with the best
  proper-noun recall on every file. Revised the article's tables, MAI and
  Scribe sections, surprises, recommendations and caveats on top of the user's
  own edits; updated the results document and the issues log (items 15, 18,
  18a). Lesson for the method: the single-file Scribe result had been
  representative of its rank but not of its margin, which only the earnings
  calls revealed.

## 7. Resuming the local half (19:46 onward)

- Relaunched the same command from the persisted copies; the script skipped the
  22 finished runs and started Whisper large-v3-turbo. A log monitor reports
  each completed run.
- Full test suite run after the Gemini change: 585 passed (plus the 12 new
  Gemini tests in a later run: 40 in that module).

## Files

- `tools/engine_bench.py`, `tools/bench_refs.py`: the benchmark and scorer.
- `%APPDATA%\TranscriberStudio\bench\bench_public\report.md`: the scored
  report; `bench_private\`: the first run on the user's own recordings;
  `bench_gemini_fix*\`: Gemini re-runs; `gemini_raw\`: the captured response.
- `docs/benchmark-issues-log.md`: every problem hit, with numbers.
