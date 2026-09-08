# Issues log: building and benchmarking four speech engines

Running notes for the comparison article. Everything here was observed while
building Transcriber Studio's engine integrations or while running the
benchmark on 2026-09-04 and 2026-09-05. Numbers are from the benchmark's own
report unless a source is named. Items marked "open" are not yet root-caused.

## Benchmark setup, for reference

- Public recordings with reference transcripts: AMI ES2004b (39 min, four-person
  meeting; table-array mic and headset mix), AMI EN2002a (36 min, table-array
  mic), Earnings-22 calls 4474506 (Costco, US, 66 min) and 4483937 (UK, 61 min).
  AMI audio transcoded to 16 kHz mono MP3 at 56 kbps to match Plaud exports.
- Scoring: word error rate after Whisper's English text normaliser (the Open ASR
  Leaderboard method), proper-noun recall, speaker attribution accuracy on
  aligned words after an optimal speaker mapping, diarization error rate via
  pyannote.metrics (0.25 s collar), and words emitted during reference silence.
- No glossary hints were sent for the public recordings.
- ElevenLabs Scribe ran on one recording only because of its token quota (see
  below).

## MAI-Transcribe-2 (Azure Speech, enhanced mode, public preview)

1. **Phrase list capped at 50, undocumented.** A 64-term glossary made every
   request fail with HTTP 400 "Context list cannot have more than 50 items".
   Found by bisecting. The app now truncates and logs how many terms were left
   out (17 on the benchmark runs).
2. **"Clean" style drops real content, not just fillers.** On the user's own
   25-second clip: 74 words to 61, a whole clause gone. On the public set, clean
   was worse than verbatim on every recording, by 1.8 to 5.0 WER points
   (ES2004b table mic 13.7 to 16.8; EN2002a 26.4 to 31.4; Costco 6.9 to 10.7;
   UK 8.0 to 9.8; ES2004b headset 11.8 to 15.0). The normaliser already
   discounts fillers, so the gap is words.
3. **Confidence field is always zero.** The response schema has a per-phrase
   confidence; the model never fills it.
4. **Speaker diarization fails on anything past about 15 minutes.** 25-second
   clip: OK. 15 min: HTTP 408 "Timeout" three times. 30 min: HTTP 503 wrapping
   "Diarization service returned error code 400". 73 min: 408, 500, 503. The
   same files transcribed fine with diarization off, including the 73-minute,
   70 MB one in a single request. A 17.5 MB re-encode failed the same way, so
   size is not the cause. The errors read like network faults and no documented
   limit predicts them (quotas page says 5 h, REST reference says 2 h, MAI page
   says 300 MB and no duration). Reported via a Q&A thread and docs PR
   MicrosoftDocs/azure-ai-docs#835.
5. **No route around it inside Azure.** The batch transcription API accepts blob
   URLs but silently drops `enhancedMode` and runs a different model; the fast
   endpoint refuses `audioUrl` when a MAI model is named. The audio has to go
   up in the request body.
6. **Upload timeouts were a client problem.** urllib3 applies the connect
   timeout to writing the request body, so a 70 MB upload died with "The write
   operation timed out". Fixed with a size-based timeout budget and streaming
   upload progress. Before that the app gave no feedback during a long upload.
7. **Resolution in the app.** MAI now runs as a decoder only, with pyannote
   separating speakers locally on its word timings. That works at any length
   and returns the voice embeddings voiceprints need.

## Gemini 3.5 Transcribe

8. **Two modes, one usable.** "Smart" returns prose and the API refuses both
   speaker labels and timestamps in that mode. Only "verbatim" fills the app's
   data model.
9. **Length limits force splitting.** 60 minutes per request, 30 with speaker
   labels. The app cuts at silence with a 3-minute overlap and matches speaker
   numbers across the seam by the repeated speech. Machinery that had to be
   built, and a place things can go wrong.
10. **No custom vocabulary alongside speakers or timestamps.** Gemini gets none
    of the glossary the other engines use.
11. **Word timestamps are unreliable.** Across the four benchmark runs, 25
    segments ended before they started and one word in the Costco call started
    at 99,711 seconds into a 66-minute file. These occur inside the first part
    too, so they are the model's, not the app's split offsets. Left alone they
    made the speaker timeline fiction: diarization error 5,436 percent and
    "speech coverage" 50 times the recording length on the Costco call. The
    app now judges each word's start against the median of its neighbours and
    re-places outliers; the first version of that repair, which chained on the
    previous word instead, stacked 3,447 words at the one absurd instant.
    Repairs logged on the re-run: 27 in part 1, 3,799 in part 2 of the Costco
    call (almost all of them collateral from that single outlier).
12. **Decoding loops.** On the Costco call Gemini repeated a seventeen-word
    sentence ("I mean, it used to be that we talk about when we had 400
    warehouses and the average") 60 times in a row inside one segment, about
    1,800 inserted words, and WER of 38.0 percent against MAI's 6.9. EN2002a
    had 1,134 repeated eight-word-window instances against MAI's zero and WER
    of 43.2 percent against 26.4. The UK call, where nothing looped, scored
    8.5 against MAI's 8.0. The app now collapses a run of five or more words
    repeated three or more times consecutively (window up to 40 words,
    punctuation ignored) and logs it. Re-scored results pending.
12a. **Root cause of 11 and 12: requests longer than the documented 30 minutes.**
    The same 30-minute stretch of the Costco call, cut with ffmpeg and sent to
    Gemini three times on its own, came back clean every time (5,132, 5,132 and
    5,137 words; no timing faults; no loop). Inside the app that stretch was
    part of a 33-minute request, because the splitter sized parts at 30
    minutes seam to seam and then prepended the 3-minute overlap. Both app runs
    of that 33-minute part failed the same way. EN2002a (35.7 min) and ES2004b
    (39 min) went as single requests because an earlier probe had found the API
    *accepts* up to about 54 minutes with speakers and word timings; it does,
    but what it returns past 30 is not sane. The UK call's 33-minute part
    happened to come back clean, so the degradation is probabilistic, rising
    with length past 30. Fix: the split threshold is now the documented 30
    minutes and no part exceeds it lead-in included (seams 27 minutes apart).
    Re-scored with the same model and the same audio:

    | Recording | Gemini before | timing repair only | + 30-min parts | MAI verbatim |
    |---|---|---|---|---|
    | Costco call WER | 38.0% | 30.1% | 7.8% | 6.9% |
    | Costco speaker attribution | 0.51 | 0.51 | 0.98 | 0.99 |
    | Costco diarization error | 5,436% | 56% | 6.3% | 5.8% |
    | Costco proper-noun recall | 0.80 | 0.80 | 0.93 | 0.92 |
    | EN2002a WER | 43.2% | 43.1% | 33.4% | 26.4% |
    | EN2002a proper-noun recall | 0.51 | 0.51 | 0.65 | 0.72 |
    | ES2004b table mic WER | 17.0% | not run | 17.3% | 13.7% |
    | ES2004b diarization error | 1,885% | not run | 50% | 20% |

    Timing repairs per part fell from thousands to 1 to 30. The general
    lesson: "accepted" and "transcribed correctly" are different limits, and
    only a reference transcript tells them apart.
12b. **Splitting has its own cost: speakers across the join.** With ES2004b now
    sent as two parts, WER was unchanged but speaker attribution fell from 0.72
    to 0.59 because one voice could not be matched across the join and became
    an extra speaker. On the Costco call, 14 voices could not be matched, which
    is expected: analysts on an earnings call each speak once, so nobody is
    present on both sides of a seam. The overlap-and-match design only works
    when the same people talk on both sides. The same cost showed on the two
    recordings that had been fine as single requests: UK call attribution 0.79
    to 0.73 (4 speakers found became 6), ES2004b headset mix 0.98 to 0.78, with
    WER unchanged on both (8.5 and 15.0 percent). So for 30-to-54-minute
    recordings there is a real trade: one request risks the collapse seen on
    EN2002a; two parts guarantee sane timings but may split a speaker in two.
    The app takes the safe side. MAI with local pyannote has no such trade
    because nothing is split.
13. **Speaker attribution after the fix is still weak where loops occurred.**
    0.51 on Costco (18 speakers found, 17 in reference) and 0.68 on EN2002a
    (6 found, 4 in reference), against 0.94 to 1.00 for MAI with pyannote.
    Open: how much of this is the loops and how much the seam bridging.
14. **As an LLM for cleanup, Gemini Flash spent its output budget thinking.**
    One real batch burned 15,728 of a 16,384-token budget on reasoning and
    truncated the JSON. Fixed by `thinkingBudget: 0` for that task, with a
    fallback when a model rejects the parameter. Until measured, the failure
    looked like batches that were too big.

14a. **Gemini's usage block under-reports output.** `total_output_tokens` came
    back 0 on every transcription response while the detailed
    `model_invocation_token_counts` block showed about 6,400 text tokens per
    30-minute part. The app logs `total_tokens`, so its "tokens billed" line
    counts audio only (about 1,500 audio tokens per minute, plus the 3-minute
    overlaps on split files). Priced from Google's page ($2 per million audio
    tokens, $12 per million text tokens), the runs here cost about $0.33 per
    hour of audio against Google's blended estimate of $0.30.

## ElevenLabs Scribe

15. **Token quota constrains testing.** About 330 tokens per minute of audio
    against a monthly allowance, so an hour of audio is a large share of the
    month. The benchmark first ran Scribe on one 39-minute meeting (about
    12,900 tokens) plus the earlier 20-minute personal recording; the user
    then chose to spend credits and it ran on all five recordings, roughly
    80,000 tokens for the four hours.
16. **No voice embeddings**, so speakers cannot be matched across recordings
    and voiceprints cannot name anyone.
17. **No seam to intervene at.** One call does everything; no vocabulary hints
    the way Whisper takes hotwords, no separate diarization step to tune.
18. **Where it did well, on all five recordings.** Second to MAI everywhere:
    14.9 vs 13.7, 12.6 vs 11.8, 27.4 vs 26.4, 7.0 vs 6.9, 8.2 vs 8.0. Best
    proper-noun recall on every file (0.82, 0.82, 0.75, 0.93, 0.87). Fewest
    deletions and most insertions of any engine on every file: it transcribes
    more of the overlapped speech and gets more of it wrong (EN2002a: 1,075
    deletions and 831 substitutions against MAI's 1,557 and 442). Found the
    correct 4 speakers on the ES2004b headset mix where pyannote found 6; found
    6 for 4 on the UK call. Attribution 0.96 / 0.97 / 0.81 / 0.99 / 0.98; DER
    two to ten points behind pyannote on four files. Per-segment confidence
    means 0.95 to 0.99 with 0 to 4 low segments per file. Fastest engine by
    far: 30 to 109 times realtime, the 61-minute UK call back in 34 seconds.
    Words during reference silence 87 on ES2004b table mic (MAI 64) but 9 on
    EN2002a, so not systematic. The earlier single-run conclusion understated
    it: the one-file result was representative, but the earnings-call tie with
    MAI only showed once it ran there.
18a. **Correction to an earlier note.** Scribe is the only *cloud* engine
    returning a usable confidence. Whisper reports a per-segment confidence
    too (EN2002a: mean 0.76, 47 of 400 segments low).

## Whisper (faster-whisper, local)

19. **Batched pipeline ignores the hallucination-silence threshold.**
    `BatchedInferencePipeline` hardcodes `hallucination_silence_threshold=None`
    whatever is passed. Found only by reading the installed source.
20. **Hallucination on bad audio.** Whisper's failure mode is fluent invented
    text. The app disables carry-over context between windows and watches for
    speech-free stretches inside "speech" segments; it limits but does not
    eliminate it.
21. **Large-v3-turbo scored far behind the cloud engines** on the first three
    public recordings: 21.0 against MAI 13.7 (ES2004b table mic), 39.6 against
    26.4 (EN2002a), 11.1 against 6.9 (Costco), each time with 10 to 20 percent
    fewer words than MAI produced. That pattern is dropped speech rather than
    mishearing. On the UK earnings call, clean audio with one speaker at a
    time, it was close: 9.9 against 8.0.
    Where the words go, on ES2004b (table mic): the losses are spread evenly
    over time (108 to 240 missed or changed reference words per five-minute
    bucket, against 47 to 152 for MAI) and over all four speakers (10 to 21
    percent of each speaker's words deleted, worst for the quietest speaker).
    The VAD reported cutting 4.7 of 39.1 minutes as non-speech, on a meeting
    where 248 of the 487 reference utterances are under 1.5 seconds
    (backchannels: "Yeah", "Mm-hmm", "Okay"). So the front end is discarding
    short, quiet utterances before the decoder sees them, on top of whatever
    the four-layer decoder costs.
    Apportioned: large-v3 with VAD on scored the same as turbo on every file
    (23.0 / 19.8 / 39.3 / 11.3 / 10.1 against turbo's 21.0 / 20.7 / 39.6 /
    11.1 / 9.9), so decoder size is not it. Turbo with VAD off recovered about
    one point (21.0 to 19.9 on ES2004b, 39.6 to 39.0 on EN2002a) and about 140
    words per file; deletions stayed at 953 and 2,514 against MAI's 568 and
    1,557 on the same audio. So the VAD is a minor contributor and the bulk is
    Whisper itself: it decodes one voice per 30-second window and, in the
    overlapping, backchannel-heavy stretches that the AMI reference transcribes
    from four headsets, it leaves the second voice out. On the earnings calls,
    where one person speaks at a time, Whisper is within 2 to 4 points of MAI.
    Per-speaker attribution and DER for Whisper are as good as MAI's, because
    the same pyannote stage runs on both.
22. **Setup hurdle.** Speaker separation needs a HuggingFace token and accepting
    terms on three gated models. Large-v3 in float16 wants about 10 GB of VRAM.
23. **Speed.** An hour of audio is many minutes of GPU work and slower than
    realtime on CPU; CrisperWhisper roughly realtime even on a GPU.

## pyannote speaker separation (used with Whisper and MAI)

24. **Speaker counting.** Found 6 speakers where AMI ES2004b has 4, on both the
    table mic and the headset mix; found 15 where the Costco call has 17; got
    EN2002a and the UK call right. Attribution on aligned words was still
    0.94 to 1.00, and diarization error 6 percent on the earnings calls and 17
    to 20 percent on the AMI table mic, so the extra speakers carry little
    speech.
25. **Voice data lost in transit.** Voiceprint embeddings were dropped when
    results were copied, queued, or resumed; fixed in the queue store, resume
    store and job copy.
26. **MAI spacing entries crashed speaker assignment.** MAI interleaves spacing
    tokens with no timing; the assignment raised `KeyError('start')` and every
    speaker was silently lost. Now skipped.
27. **Orphan words in pauses had no speaker**; now assigned to the nearest turn
    within 2 seconds.

## Plaud

28. **Official API is read-only.** Renaming a recording from the app needed
    Plaud's private web API (`PATCH /file/{id}`), with its own token that is
    not the CLI's, a `typ` header distinguishing token kinds, a status -302
    for region mismatch, and -420 for a retired refresh token that presented
    in the app as a silent logout followed by a crash on Logout (a QThread
    destroyed while running). Both fixed.

## Benchmark methodology

29. **No ground truth for the user's own recordings.** The first run could only
    measure agreement between engines; that is why public corpora were used.
30. **Synthetic TTS test considered and deferred.** Feasible with the existing
    Azure key (196 English neural voices), but neural TTS is cleaner than real
    speech and a mispronounced name would fail every engine for the wrong
    reason. The public sets came first.
31. **Proper-noun heuristic is approximate.** AMI capitalises the first word of
    every utterance and stores punctuation as separate tokens, so a first pass
    counted "Yeah" and "Okay" as names. Sentence-initial words are now flagged
    at parse time; a few still leak through. Treat the recall numbers as
    directional.
32. **Normalising word by word broke WER.** "twenty two" became "20 2" against
    an engine's "22". Headline WER is now computed on whole normalised texts;
    per-token alignment is used only where the reference word must be known.
33. **Data access friction.** Earnings-22 audio is behind Git LFS with the
    subset directory holding symlinks, so the first download produced empty
    files. DiPCo is a single 13.4 GB archive and was skipped. The OneDrive
    Downloads folder stalled a plain directory listing (files on demand).
34. **A cost cap and a hold.** Scribe was capped by duration and later by
    recording name; the local Whisper half was held with a watcher that killed
    the run at the first Whisper start so the laptop could be shut down, then
    resumed from the saved results.
