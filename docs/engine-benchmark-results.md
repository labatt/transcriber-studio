# Speech engine benchmark: results

Four engines, five recordings with reference transcripts, scored on 2026-09-05
with `tools/engine_bench.py`. Method and every problem hit along the way are in
`benchmark-process-log.md` and `benchmark-issues-log.md`. The raw report with
every attribute is `engine-benchmark-report.md`. The narrative version is
`article-stt-engine-benchmark.md`.

## What was run

| Recording | Source | Length | Speakers | Character |
|---|---|---|---|---|
| ES2004b, table-array mic | AMI Meeting Corpus | 39.1 min | 4 | Scripted design meeting, distant mic, non-native English |
| ES2004b, headset mix | AMI Meeting Corpus | 39.1 min | 4 | Same meeting, close mics (microphone control) |
| EN2002a, table-array mic | AMI Meeting Corpus | 35.7 min | 4 | Unscripted meeting, heavy overlap and cross-talk |
| Costco earnings call (4474506) | Earnings-22 | 66.3 min | 17 | US English, prepared remarks then Q&A |
| UK earnings call (4483937) | Earnings-22 | 60.9 min | 4 | British English, same domain |

AMI audio was transcoded to 16 kHz mono MP3 at 56 kbps, the format a Plaud
export uses. Earnings-22 audio is already low-bitrate MP3. No glossary hints
were sent. Every run went through the app's own transcribe path; MAI and
Whisper used local pyannote for speakers, Scribe and Gemini their own.

Engines: MAI-Transcribe-2 verbatim and clean (Azure Speech, public preview),
ElevenLabs Scribe, Gemini 3.5 Transcribe verbatim, Whisper large-v3-turbo and
large-v3 (faster-whisper, RTX 2000 Ada laptop GPU). Scribe ran on all five
recordings, about 80,000 tokens at its 330-tokens-per-minute rate.

Gemini appears twice: as first run, and after two fixes the benchmark forced
(see "What broke" below). The fixed column is the one to compare.

## Word error rate

Lower is better. Scored after Whisper's English text normaliser, so fillers,
punctuation, case and number formatting do not count. Reference word counts:
6,819 / 6,819 / 7,846 / 11,239 / 8,427.

| Recording | MAI verbatim | Scribe | Gemini (fixed) | MAI clean | Whisper turbo | Whisper large-v3 | Gemini (first run) |
|---|---|---|---|---|---|---|---|
| ES2004b table mic | **13.7** | 14.9 | 17.3 | 16.8 | 21.0 | 23.0 | 17.0 |
| ES2004b headset | **11.8** | 12.6 | 15.0 | 15.0 | 20.7 | 19.8 | 15.3 |
| EN2002a table mic | **26.4** | 27.4 | 33.4 | 31.4 | 39.6 | 39.3 | 43.2 |
| Costco call | **6.9** | 7.0 | 7.8 | 10.7 | 11.1 | 11.3 | 38.0 |
| UK call | **8.0** | 8.2 | 8.5 | 9.8 | 9.9 | 10.1 | 8.5 |

## Error composition, MAI verbatim against Scribe

The two leaders reach similar totals by opposite routes. Scribe deletes the
fewest reference words of any engine on every file and inserts the most; MAI
substitutes far less.

| Recording | MAI deletions | Scribe deletions | MAI substitutions | Scribe substitutions | MAI insertions | Scribe insertions |
|---|---|---|---|---|---|---|
| ES2004b table mic | 568 | 403 | 255 | 377 | 110 | 238 |
| ES2004b headset | 462 | 320 | 236 | 310 | 108 | 228 |
| EN2002a table mic | 1,557 | 1,075 | 442 | 831 | 71 | 246 |
| Costco call | 79 | 55 | 229 | 223 | 465 | 509 |
| UK call | 105 | 83 | 209 | 214 | 357 | 393 |

## Proper-noun recall

Share of capitalised, non-sentence-initial reference words that came through
intact. Approximate: a few sentence-initial words leak into the AMI name lists.

| Recording | MAI verbatim | MAI clean | Scribe | Gemini (fixed) | Whisper turbo | Whisper large-v3 |
|---|---|---|---|---|---|---|
| ES2004b table mic | 0.74 | 0.74 | **0.82** | 0.71 | 0.62 | 0.62 |
| ES2004b headset | 0.79 | 0.77 | **0.82** | 0.71 | 0.62 | 0.62 |
| EN2002a table mic | 0.72 | 0.61 | **0.75** | 0.65 | 0.52 | 0.49 |
| Costco call | 0.92 | 0.91 | **0.93** | **0.93** | 0.80 | 0.79 |
| UK call | 0.85 | 0.84 | **0.87** | 0.85 | 0.82 | 0.83 |

Names no engine got, for a sense of the ceiling: Buccleuch, Heinbockel, Melich,
Hawkline, Hurn, Meggitt, Renishaw, Videoplus.

## Speakers

Speaker attribution is the share of correctly transcribed words assigned to the
right person after the best one-to-one mapping of labels. Diarization error
rate (DER) is pyannote.metrics with a 0.25 s collar; it penalises missed
speech, false speech and confusion together.

| Recording | Metric | MAI + pyannote | Scribe | Gemini (fixed) | Whisper + pyannote |
|---|---|---|---|---|---|
| ES2004b table mic (4 spk) | speakers found | 6 | 6 | 5 | 6 |
| | attribution | 0.94 | **0.96** | 0.59 | **0.96** |
| | DER | **0.20** | 0.25 | 0.50 | 0.22 |
| ES2004b headset (4 spk) | speakers found | 6 | **4** | **4** | 6 |
| | attribution | 0.95 | **0.97** | 0.78 | **0.97** |
| | DER | **0.17** | 0.21 | 0.33 | 0.21 |
| EN2002a table mic (4 spk) | speakers found | 4 | 5 | 5 | 4 |
| | attribution | 0.87 | 0.81 | 0.70 | **0.89** |
| | DER | **0.40** | 0.50 | 0.54 | 0.45 |
| Costco call (17 spk) | speakers found | 15 | 15 | 18 | 15 |
| | attribution | **0.99** | **0.99** | 0.98 | **0.99** |
| | DER | 0.06 | **0.06** | 0.06 | 0.07 |
| UK call (4 spk) | speakers found | 4 | 6 | 6 | 4 |
| | attribution | **1.00** | 0.98 | 0.73 | **1.00** |
| | DER | **0.06** | 0.08 | 0.32 | 0.07 |

MAI and Whisper share the pyannote numbers because the same speaker stage runs
on both; the small differences come from the words each decoder placed.

## Confidence

Only two engines return a per-segment confidence that means anything.

| Recording | Scribe mean (segments, low) | Whisper turbo mean (segments, low) |
|---|---|---|
| ES2004b table mic | 0.96 (556, 3) | not extracted |
| ES2004b headset | 0.98 (576, 2) | not extracted |
| EN2002a table mic | 0.95 (909, 4) | 0.76 (400, 47) |
| Costco call | 0.99 (236, 0) | not extracted |
| UK call | 0.99 (138, 0) | not extracted |

MAI's confidence field is present and always zero. Gemini has none.

## Speed

Audio minutes transcribed per wall-clock minute, end to end including upload,
speaker separation and the app's own overhead. Read with care: the second and
later local-pipeline runs on the same file (MAI clean, both Whisper models)
hit the app's diarization cache from the MAI verbatim run, so their figures
exclude pyannote's cost, and the first Whisper run includes model loading.

| Recording | MAI verbatim (cold pyannote) | Scribe | Gemini (fixed) | Whisper turbo (cached pyannote) | Whisper large-v3 (cached pyannote) |
|---|---|---|---|---|---|
| ES2004b table mic | 14× | 30× | 24× | 11× | 10× |
| ES2004b headset | 13× | 77× | 22× | 12× | 12× |
| EN2002a table mic | 15× | 63× | 19× | 11× | 10× |
| Costco call | 15× | 90× | 22× | 11× | 10× |
| UK call | 14× | 109× | 26× | 13× | 12× |

Scribe returned every file in 31 to 79 seconds. On the laptop GPU, Whisper
decoding alone runs at roughly 10 to 13 times realtime.

## Findings

1. **MAI verbatim had the lowest word error rate on every recording, and
   Scribe was second on every recording, 0.1 to 1.2 points behind.** On the two
   earnings calls the gap is a tenth or two of a point, which is a tie. Over
   Gemini and Whisper MAI's lead is one to thirteen points, largest on the
   meetings.
2. **Scribe had the best proper-noun recall on all five files.** It also
   deleted the fewest reference words and inserted the most on every file: it
   transcribes more of the overlapped speech and is wrong about more of what it
   transcribes, and the two nearly cancel against MAI's more conservative
   output.
3. **MAI clean costs 2 to 5 points of WER everywhere.** The normaliser already
   discounts fillers, so that is content. Use verbatim and let a cleanup pass
   tidy it where the changes can be seen.
4. **Gemini's first-run numbers were an app problem, not a model problem.**
   Requests over the documented 30 minutes came back with broken word timings
   and, on one call, a 17-word sentence repeated 60 times. Capping every
   request at 30 minutes (lead-in included) took the Costco call from 38.0 to
   7.8 percent WER and speaker attribution from 0.51 to 0.98. Splitting has its
   own cost: a voice not present on both sides of a seam becomes an extra
   speaker, which is why Gemini's attribution trails on the recordings that
   used to fit in one request.
5. **Whisper's gap is dropped speech, not mishearing, and it is Whisper
   itself.** Turbo and large-v3 score within two points of each other on every
   file and produce the same word counts, 10 to 20 percent below MAI on the
   meetings, so decoder size is not the variable. A diagnostic run with the VAD
   front end off recovered only about one point and 140 words; Whisper still
   deleted 950 and 2,500 reference words where MAI deleted 570 and 1,560.
   Whisper decodes one voice per window and leaves the second voice out. On the
   earnings calls, one speaker at a time, it is within 2 to 4 points of MAI.
6. **Microphone distance mattered less than expected.** Headset mics improved
   MAI by two points, Scribe by two, Gemini by two; Whisper barely moved.
7. **Speaker counting.** pyannote over-counted the scripted AMI meeting (6 for
   4, on both mic setups) yet attributed words at 0.94 to 0.97; Scribe got the
   headset mix right at 4 and matched pyannote's 6 on the table mic. Both
   under-counted the 17-person earnings call (15) with attribution 0.99. Scribe
   split the UK call's four people into six, at attribution 0.98. pyannote's
   DER was best or tied on all five files.
8. **Scribe is the fastest engine by a wide margin** (30 to 109 times
   realtime) and the only cloud engine with a usable confidence. Its costs are
   the token quota and that the audio leaves the machine; it returns no voice
   embeddings, so speakers cannot be matched across recordings.

## Cost

| Engine | Per hour of audio | This benchmark (4 h) | Basis |
|---|---|---|---|
| MAI-Transcribe-2 | $0.10 | about $0.40 | promotional rate through end of 2026 |
| ElevenLabs Scribe | $0.40 (starter plan), $0.22 (other plans) | about $1.60 or $0.90 | per-plan rates supplied by the user; token allowance plans bill about 330 tokens/min, roughly 80,000 tokens here |
| Gemini 3.5 Transcribe | about $0.33 measured; $0.30 by Google's blended estimate ($0.005/min) | about $1.30, plus re-runs | Google pricing page: $2/M audio input tokens, $12/M text output tokens. Logged usage: 63,140 tokens for a 39-min file split in two, 108,402 for 66 min in three parts (about 1,500 audio tokens/min plus 3-min overlaps); text output about 6,400 tokens per 30-min part |
| Whisper | $0 marginal | $0 | laptop GPU; about 6 min of compute per hour of audio |

Note: the app logs the API's `total_tokens`, which on these responses counted
audio input only (`total_output_tokens` came back 0 while the detailed block
showed about 6,400 text tokens per part). The measured Gemini figure above
adds the text tokens from the detailed block.

## Caveats

- Five recordings is a small sample. The ranking is consistent across them,
  the exact margins are not to be quoted to a decimal, and the top two are a
  tenth of a point apart on the earnings calls.
- No accents beyond US, UK and the AMI participants' non-native English.
- The proper-noun metric is a heuristic; the speaker metrics depend on the
  reference's own segmentation, which for AMI includes overlapping speech that
  no engine here transcribes twice.
- Speed figures mix cold and cached speaker separation, as noted.
