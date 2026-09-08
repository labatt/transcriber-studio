# Four speech-to-text engines, scored against transcripts we already had

So what's the best mainstream speech to text transcription engine?

I own a Plaud recorder. It makes exactly the kind of recordings transcription tools handle worst: a mic on a meeting table, someone talking from across the room, a phone call on speaker. Plaud will transcribe those for a monthly fee, and I got tired of paying to transcribe audio I had already made on hardware I had already bought. So I built my own transcription app and, while building it, wired in four speech engines: OpenAI's Whisper running locally, ElevenLabs Scribe, Google's Gemini 3.5 Transcribe, and Microsoft's new MAI-Transcribe-2 through Azure Speech.

The app is Transcriber Studio: https://github.com/labatt/transcriber-studio. It is open source under the GPL, runs on your own machine, imports straight from a Plaud account or takes any audio file, and lets you pick the engine per run. That last part is what made a fair comparison possible. The same recordings go through the same pipeline, and only the decoder changes.

I started by writing up my impressions of the four engines, which is pretty much all you can write when you have no reference transcripts to check against. But then I ran all four against public recordings that come with human transcripts, looked up how the speech research community scores things, and kept a log of everything that broke and how they broke and how I compensated. The log turned out to be at least as useful as the scores, because one of the engines was being badly misjudged by my own code and only a reference transcript could show it.

## Why the first attempt was not good enough

My first benchmark used two of my own meetings. I ran every engine over them and measured what I could without a reference: speed, word counts, speaker counts, how many glossary terms each engine found, fillers kept or dropped, repeated phrases, and how far each engine's words were from every other engine's. That last one is a word error rate between engines, and it says how much they disagree, not who is right. Four engines can definitely agree on the wrong word. I didn't have an actual accurate transcript for those meetings so I couldn't compare the produced transcripts from the actual transcripts. 

So I stopped that run and went looking for recordings with fully reviewed and accurate transcripts.

## The recordings

Two corpora fit the kind of thing I record.

The AMI Meeting Corpus (https://groups.inf.ed.ac.uk/ami/corpus/) is a hundred hours of real four-person meetings recorded in three rooms, on headsets and on microphone arrays sitting on the table, with word-level transcripts and speaker labels. It is released under CC BY 4.0. The table-array channel is the closest public thing to a Plaud on a conference table. I used two meetings: ES2004b, a scripted product design meeting (39 minutes), and EN2002a, an unscripted meeting with a lot of overlap and cross-talk (36 minutes). For ES2004b I also ran the headset mix, so the microphone effect could be measured on identical speech.

Earnings-22, from Rev (https://github.com/revdotcom/speech-datasets), is 119 hours of earnings calls with transcripts, speaker labels and timestamps, full of company names, tickers and numbers, across many accents. I used two calls from its ten-file subset: Costco's (66 minutes, US English, 17 speakers) and a UK company's (61 minutes, 4 speakers). The transcripts are CC BY-SA 4.0.

I transcoded the AMI audio to 16 kHz mono MP3 at 56 kbps, which is what a Plaud export looks like, so the engines heard the kind of file my recorder produces. The Earnings-22 audio is already low-bitrate MP3.

That is five recordings and four hours of audio, which is a small sample. I'll come back to what that means.

## How I scored

I used Claude Code to run all of the tests and then assess the scores.

**Word error rate**, after running both the reference and each engine's output through the English text normaliser that ships with Whisper. The Hugging Face Open ASR Leaderboard scores with the same normaliser. It lowercases, strips punctuation, turns number words into digits and removes fillers like "um" and "uh", so an engine is not punished for writing "22" where the transcript says "twenty two", and a verbatim engine is not punished for keeping the ums.

**Proper-noun recall:** the share of capitalised, non-sentence-initial reference words that came through intact. This is a heuristic and it leaks. AMI capitalises the first word of every single, well, word, and stores punctuation as separate tokens, so my first pass counted "Yeah" and "Okay" as names until I flagged sentence starts at parse time. A few still slip through. Treat the recall numbers as directional.

**Speaker attribution:** of the words an engine got right, what share it gave to the right person, after the best one-to-one mapping of its labels to the reference's. And diarization error rate from the pyannote.metrics library with a quarter-second collar, which penalises missed speech, false speech and confusion together.

**Words during reference silence:** hypothesis words that land where the reference has nobody speaking for a second either side. That's a hallucination.

Claude checked the scorer by feeding it a transcript built from the reference. Zero word error rate, attribution of 1.0, zero diarization error. That step caught two bugs in the scoring and prevented any engine getting blamed for them.

No glossary hints were sent to any engine. I have a vocabulary of names I've generated but it has nothing to do with Costco or a role-played design team.

One cost note. ElevenLabs bills Scribe at about 330 tokens per minute of audio against a 30,000-token monthly allowance, so an hour is more than half the month. I nearly left it at one meeting for that reason, then ran it on everything out of credits I had. All four hours came to roughly 80,000 tokens, which is the one line in this benchmark that would stop me repeating it casually.

## The results

Word error rate, lower is better. Reference word counts are 6,819 for each ES2004b mix, 7,846 for EN2002a, 11,239 for the Costco call and 8,427 for the UK call.

| Recording                | MAI verbatim | MAI clean | Scribe  | Gemini (after fix) | Gemini (first run) | Whisper turbo | Whisper large-v3 |
| ------------------------ | ------------ | --------- | ------- | ------------------ | ------------------ | ------------- | ---------------- |
| AMI ES2004b, table mic   | 13.7%        | 16.8%     | 14.9%   | 17.3%              | 17.0%              | 21.0%         | 23.0%            |
| AMI ES2004b, headset mix | 11.8%        | 15.0%     | 12.6%   | 15.0%              | 15.3%              | 20.7%         | 19.8%            |
| AMI EN2002a, table mic   | 26.4%        | 31.4%     | 27.4%   | 33.4%              | 43.2%              | 39.6%         | 39.3%            |
| Costco earnings call     | 6.9%         | 10.7%     | 7.0%    | 7.8%               | 38.0%              | 11.1%         | 11.3%            |
| UK earnings call         | 8.0%         | 9.8%      | 8.2%    | 8.5%               | 8.5%               | 9.9%          | 10.1%            |

Gemini appears twice because the benchmark found a bug in how my app was using it. The "after fix" column is the honest comparison, and the story of the "first run" column is below.

Proper-noun recall, higher is better.

| Recording            | MAI verbatim | MAI clean | Scribe  | Gemini (after fix) | Whisper turbo | Whisper large-v3 |
| -------------------- | ------------ | --------- | ------- | ------------------ | ------------- | ---------------- |
| ES2004b, table mic   | 0.74         | 0.74      | 0.82    | 0.71               | 0.62          | 0.62             |
| ES2004b, headset mix | 0.79         | 0.77      | 0.82    | 0.71               | 0.62          | 0.62             |
| EN2002a, table mic   | 0.72         | 0.61      | 0.75    | 0.65               | 0.52          | 0.49             |
| Costco call          | 0.92         | 0.91      | 0.93    | 0.93               | 0.80          | 0.79             |
| UK call              | 0.85         | 0.84      | 0.87    | 0.85               | 0.82          | 0.83             |

Names nobody got, to show where the ceiling is: Buccleuch, Heinbockel, Melich, Hawkline, Hurn, Meggitt, Renishaw, Videoplus.

The speaker table puts MAI and Whisper in one column, because in my app the same local pyannote stage separates speakers for both. The small differences between them come from the words each decoder placed.

| Recording                       | Metric            | MAI or Whisper with pyannote | Scribe  | Gemini (after fix) |
| ------------------------------- | ----------------- | ---------------------------- | ------- | ------------------ |
| ES2004b, table mic (4 speakers) | speakers found    | 6                            | 6       | 5                  |
|                                 | attribution       | 0.94 to 0.96                 | 0.96    | 0.59               |
|                                 | diarization error | 0.20 to 0.22                 | 0.25    | 0.50               |
| ES2004b, headset (4)            | speakers found    | 6                            | 4       | 4                  |
|                                 | attribution       | 0.95 to 0.97                 | 0.97    | 0.78               |
|                                 | diarization error | 0.17 to 0.21                 | 0.21    | 0.33               |
| EN2002a, table mic (4)          | speakers found    | 4                            | 5       | 5                  |
|                                 | attribution       | 0.87 to 0.89                 | 0.81    | 0.70               |
|                                 | diarization error | 0.40 to 0.45                 | 0.50    | 0.54               |
| Costco call (17)                | speakers found    | 15                           | 15      | 18                 |
|                                 | attribution       | 0.99                         | 0.99    | 0.98               |
|                                 | diarization error | 0.06 to 0.07                 | 0.06    | 0.06               |
| UK call (4)                     | speakers found    | 4                            | 6       | 6                  |
|                                 | attribution       | 1.00                         | 0.98    | 0.73               |
|                                 | diarization error | 0.06 to 0.07                 | 0.08    | 0.32               |

Speed, as audio minutes per wall-clock minute, end to end including upload and speaker separation. Read this table loosely. The second and later local runs on the same file (MAI clean and both Whisper models) reused the speaker separation cached from the first run, so their figures leave out pyannote's cost, and the first Whisper run includes loading the model from disk.

| Recording          | MAI verbatim, cold | Scribe  | Gemini (after fix) | Whisper turbo, cached | Whisper large-v3, cached |
| ------------------ | ------------------ | ------- | ------------------ | --------------------- | ------------------------ |
| ES2004b, table mic | 14×                | 30×     | 24×                | 11×                   | 10×                      |
| Costco call        | 15×                | 90×     | 22×                | 11×                   | 10×                      |
| UK call            | 14×                | 109×    | 26×                | 13×                   | 12×                      |

Scribe was the fastest engine on every file, and not by a little: the 61-minute UK call came back in 34 seconds. On the laptop GPU (an RTX 2000 Ada with 8 GB), Whisper decoding runs at ten to thirteen times realtime, so an hour of audio is five or six minutes.

Cost, per hour of audio, and what this benchmark's four hours came to.

| Engine | Per hour of audio | Four hours | Basis |
|---|---|---|---|
| MAI-Transcribe-2 | $0.10 | about $0.40 | promotional rate through the end of 2026 |
| ElevenLabs Scribe | $0.40 on the starter plan, $0.22 on other plans | about $1.60 or $0.90 | per-plan rates; on a token allowance it is about 330 tokens a minute |
| Gemini 3.5 Transcribe | about $0.33 measured, $0.30 by Google's estimate | about $1.30 | $2 per million audio tokens plus $12 per million text tokens; these runs used about 1,500 audio and 210 text tokens per minute |
| Whisper | nothing beyond the hardware | $0 | a laptop GPU, six minutes per hour of audio |

The Gemini figure moves with how much text comes back and with the 10 percent of audio that gets transcribed twice at the seams when a recording is split. Google's own blended estimate is $0.005 a minute; verbatim mode with speakers and word timings ran a little above that.

## MAI-Transcribe-2

MAI verbatim had the lowest word error rate on every one of the five recordings, but the margin depends on who is second. Over Scribe it is 0.1 to 1.2 points, which on the earnings calls is a tie. Over Gemini and Whisper it is one to thirteen points, largest on the meetings. It found 92 percent of the names in the Costco call.

Its "clean" output style cost between 1.8 and 5.0 points of word error rate on every file compared with verbatim. That surprised me less than it should have. On my own 25-second test clip earlier, clean had dropped a whole real clause, taking the clip from 74 words to 61. The normaliser already removes fillers before scoring, so the gap here is content, not ums. I run verbatim and let a language-model cleanup pass tidy it afterwards, where the changes are visible.

MAI's own speaker diarization does not appear in the tables because it does not work on anything long. With diarization enabled, a 25-second clip was fine. Fifteen minutes returned HTTP 408 "Timeout" three times. Thirty minutes returned HTTP 503 wrapping "Diarization service returned error code 400". Seventy-three minutes returned 408, then 500, then 503. The same files with diarization off transcribed perfectly, the 73-minute, 70 MB one in a single request, and a 17.5 MB re-encode of it failed the same way with diarization on, so size is not the cause. The errors read like network trouble and nothing in the documentation predicts them: the quotas page says five hours, the REST reference says two, the model's page says 300 MB and no duration. I lost real time on the wrong theory. I have a Q&A thread and a documentation pull request open with the full table of runs.

There is no way around it inside Azure. The batch transcription API accepts blob URLs and handles hours of audio, but it drops the enhanced-mode parameter without saying so and runs a different model. Passing a URL to the fast endpoint is refused the moment a MAI model is named.

The phrase list, which is how you feed it vocabulary, has a hard cap of 50 entries that the documentation does not mention. A 64-term glossary made every request fail with a 400 until I bisected it. The response also carries a per-phrase confidence field, and it is zero on every phrase.

So in my app MAI runs as a decoder only. It transcribes the whole recording in one request, and pyannote separates the speakers locally on its word timings afterwards, exactly as it does for Whisper. That is where the speaker numbers in the table come from, and it is why voiceprints work on this engine too: pyannote returns a voice embedding per speaker, which none of the cloud engines do.

At the promotional rate of ten cents per hour of audio, four hours of benchmark cost about forty cents.

## Gemini 3.5 Transcribe, and the bug that was mine

Gemini's first-run numbers were 38.0 percent on the Costco call and 43.2 on EN2002a, against MAI's 6.9 and 26.4 on the same audio. If I had published the earlier "agreement only" benchmark I would have written that Gemini was far behind. Gemini was fine. My code was far behind.

The first clue was in the transcripts. The Costco transcript had about 1,800 more words than the reference, and one eight-word window repeated 60 times inside a single 186-word segment: "I mean, it used to be that we talk about when we had 400 warehouses and the average", over and over. That is a decoding loop, the model degenerating, and the extra words were the insertions inflating the error rate. Meanwhile the speaker metrics were impossible. Diarization error of 5,436 percent, "speech coverage" fifty times the length of the recording. Twenty-five segments across the four runs ended before they started, and one word in a 66-minute file started at 99,711 seconds, about 27 hours in.

I wrote a repair step: judge each word's start against the median of its neighbours, re-place the outliers, clamp everything to the recording, and collapse a run of five or more words repeated three or more times consecutively to a single copy. The first version of the repair chained on the previous word instead of the neighbourhood and made things worse in a new way, stacking 3,447 words at the one absurd timestamp. The first version of the loop collapse missed entirely because the looped sentence was seventeen words and my window stopped at sixteen. Second version: Costco went from 38.0 to 30.1 percent and the speaker timeline became sane. Better, and still four times MAI's error rate.

Then I did what I should have done first. I cut the affected 30 minutes of the call with ffmpeg and sent it to Gemini directly, three times, saving the raw responses. All three came back clean: 5,132, 5,132 and 5,137 words, no timing faults, no loop. The model handled that audio without trouble, so whatever went wrong was in the request my app was sending.

The difference was length. Google documents a 30-minute limit per request when you ask for speaker labels or word timestamps. An earlier probe of mine had found the API accepts more: 35, 46, 51 and 54-minute requests all went through, and only 57 and 80 were refused. So I had set the app to split recordings only past 54 minutes, and sized the parts at 30 minutes seam to seam plus a 3-minute overlap at the front, for matching speakers across the join. Middle parts were 33 minutes long. EN2002a at 35.7 minutes and ES2004b at 39 went up as single requests. Every request of exactly 30 minutes came back clean. Every one over 30 came back damaged, to varying degrees: the 33-minute Costco part badly, EN2002a badly, ES2004b mildly, and the UK call's 33-minute part happened to be fine. Accepted and transcribed correctly are two different limits, and only a reference transcript tells them apart.

The fix was small. Split at the documented 30 minutes, and make 30 the maximum part length including the overlap, so seams sit 27 minutes apart. I re-ran the same model on the same audio:

|                                  | Gemini first run | timing repair only | 30-minute parts | MAI verbatim |
| -------------------------------- | ---------------- | ------------------ | --------------- | ------------ |
| Costco call, word error rate     | 38.0%            | 30.1%              | 7.8%            | 6.9%         |
| Costco call, speaker attribution | 0.51             | 0.51               | 0.98            | 0.99         |
| Costco call, diarization error   | 5,436%           | 56%                | 6.3%            | 5.8%         |
| Costco call, proper-noun recall  | 0.80             | 0.80               | 0.93            | 0.92         |
| EN2002a, word error rate         | 43.2%            | 43.1%              | 33.4%           | 26.4%        |
| EN2002a, proper-noun recall      | 0.51             | 0.51               | 0.65            | 0.72         |

Timing repairs per part fell from thousands to between one and thirty. On the Costco call Gemini went from last to second.

Splitting has its own cost, and the table shows it too. My app matches speaker numbers across a seam by transcribing the same three minutes in both parts and seeing which numbers describe the same speech. That only works when the same people talk on both sides. On an earnings call each analyst speaks once, so 14 voices could not be matched and became extra speakers (18 found for 17). On the two recordings that had fitted in one request before, splitting them cost attribution: the UK call fell from 0.79 to 0.73, the ES2004b headset mix from 0.98 to 0.78, with word error rate unchanged. For a recording between 30 and 54 minutes there is a real trade: one request risks the collapse I saw on EN2002a, two parts guarantee sane timings but may split a person in two. The app takes the safe side. MAI with local speaker separation has no such trade because nothing is split.

Gemini has two limits of its own that no fix on my side changes. Its "smart" mode returns a nicely written block of prose and refuses both speaker labels and timestamps, so only "verbatim" fills a transcript with times and speakers. And it will not take a custom vocabulary alongside either of those, so it is the one engine that gets none of my glossary.

Gemini as a language model, rather than as a transcriber, had a problem of its own. When I used a Gemini Flash model for the cleanup pass, it was spending its output budget thinking. One real batch burned 15,728 of a 16,384-token budget on reasoning and truncated the JSON it was supposed to return. Until I measured it, the failure looked like batches that were too big. Turning thinking off for a task that is mechanical restructuring fixed it.

## ElevenLabs Scribe

This is the engine I nearly left out for budget reasons, and it came second on every recording. Against MAI it was 1.2 points behind on the ES2004b table mic, 0.8 on the headset mix, 1.0 on EN2002a, and 0.1 and 0.2 on the two earnings calls, which is no difference at all. Against Gemini and Whisper it was ahead everywhere.

It gets there differently from MAI. On every file Scribe deleted the fewest reference words of any engine and inserted the most. On EN2002a, the overlap-heavy meeting, it deleted 1,075 words where MAI deleted 1,557, so it is catching more of the speech that happens under other speech, and it paid for that with 831 substitutions against MAI's 442. It transcribes more and is wrong about more of what it transcribes, and the two nearly cancel.

It had the best proper-noun recall on all five files: 0.82 and 0.82 on the two ES2004b mixes against MAI's 0.74 and 0.79, 0.75 against 0.72 on EN2002a, 0.93 on the Costco call (tied with Gemini, MAI 0.92), and 0.87 on the UK call against 0.85. If names are what you care about, this was the engine.

Speakers were good and occasionally too many. It found the correct four on the ES2004b headset mix, where pyannote said six, and attributed words at 0.97; on the table mic it said six like pyannote did, at 0.96. On EN2002a it found five and attributed at 0.81, behind MAI and Whisper with pyannote (0.87 to 0.89). On the Costco call it matched pyannote at 15 speakers for 17 with attribution 0.99. On the UK call it split four people into six, attributing at 0.98, so the extra two carried little speech. Its diarization error trailed pyannote's by two to ten points on four of the five files.

It is the only cloud engine that returns a per-segment confidence, and the numbers mean something: means of 0.95 to 0.99 across the five files, with between zero and four segments flagged low on each. Whisper reports one too (a mean of 0.76 on EN2002a with 47 segments flagged); MAI's is always zero and Gemini has none. On the ES2004b table mic it also produced the most words during reference silence (87 against MAI's 64), but only 9 on EN2002a and about the same as MAI on the earnings calls, so that is not a habit.

Architecturally it is the least work and the least control. One call transcribes and separates speakers, takes files up to 5 GB, tags non-speech events if asked, and returns in seconds. There is no seam to intervene at: no vocabulary the way Whisper takes hotwords, no separate diarization step to tune, and no voice embeddings, so speakers are numbered per recording and cannot be matched to voices you have named before. Your audio leaves the machine, and at 330 tokens a minute the monthly allowance goes quickly. Those two things, not accuracy, are what keep it from being my default.

## Whisper, and what the local pipeline actually loses

This is faster-whisper on the laptop GPU, with pyannote for speakers, the way the app ran before any of the cloud engines were in it.

Whisper large-v3-turbo scored 21.0 and 39.6 percent on the two AMI table-mic meetings, against MAI's 13.7 and 26.4, and 11.1 on the Costco call against 6.9. On the UK call it was close, 9.9 against 8.0. Each time it produced 10 to 20 percent fewer words than MAI on the meetings. That pattern is dropped speech, not misheard speech.

I had two suspects and tested both. The first was the turbo model itself, which gets its speed by cutting the decoder from 32 layers to four. So I ran large-v3, the full model. It scored the same: 23.0, 19.8, 39.3, 11.3 and 10.1, within two points of turbo on every file, with the same word counts to within a few dozen. Decoder size was not the variable.

The second suspect was the voice activity detector in front of the decoder. On the ES2004b meeting it reported cutting 4.7 of 39.1 minutes as non-speech, and half of that meeting's 487 reference utterances are under a second and a half: "Yeah", "Mm-hmm", "Okay". So I ran turbo again with the VAD off. It recovered about one point (21.0 to 19.9, 39.6 to 39.0) and about 140 words per file. Whisper still deleted 953 and 2,514 reference words where MAI deleted 568 and 1,557 on the same audio.

So the bulk of the gap is Whisper itself. It decodes one voice per thirty-second window, and in the overlapping, backchannel-heavy stretches that AMI transcribes from four separate headsets, it leaves the second voice out. The losses were spread evenly across time and across all four speakers, ten to twenty-one percent of each speaker's words, worst for the quietest. On the earnings calls, one person at a time, Whisper is within two to four points of the best engine. On meetings it is not, and no amount of model size fixes that.

The batched inference pipeline in faster-whisper hardcodes the hallucination-silence threshold to off whatever you pass, which I only found by reading the installed source. Whisper's hotwords mechanism, which re-injects your vocabulary into every decoder window, is still the best vocabulary control of the four engines in my experience, though this benchmark could not exercise it because none of my names appear in the recordings.

Whisper's speaker numbers are MAI's speaker numbers, because the same pyannote stage runs on both. pyannote found six speakers where the scripted AMI meeting has four, on both microphone setups, yet attributed words at 0.94 to 0.97, so the two extra speakers carry very little speech. It found 15 where the Costco call has 17, with attribution of 0.99: analysts who speak once get merged into a neighbour. On the earnings calls its diarization error was six to seven percent; on the AMI table mics, twenty to forty-five, the higher figure being the overlap-heavy EN2002a.

## Things I did not expect

Microphone distance mattered less than I assumed. Moving from the table array to the headset mix on the same meeting improved MAI by two points (13.7 to 11.8) and Gemini by two (17.3 to 15.0), and barely moved Whisper. I would have guessed twice that.

The scripted meeting was easy and the unscripted one was hard for everyone. EN2002a nearly doubled every engine's error rate over ES2004b, in the same kind of room on the same kind of microphone. People talking over each other cost these engines far more than distance from the mic did.

Clean-style output from a speech model is a bad idea if you care about the words. MAI's was worse than its own verbatim on every recording, and the words it drops are not fillers.

The engine I almost left out was the closest competitor. On the two earnings calls Scribe and MAI are separated by a tenth or two of a point, and Scribe found more of the names on every single file. Had I stuck to one Scribe run, the article would have understated it.

And the one I keep coming back to: an API accepting your request tells you nothing about whether the answer is right. Gemini took 54-minute requests with a smile and returned garbage timestamps and looped sentences. Only a transcript with the answers on it could show that, and my first benchmark, on my own recordings, could not have.

## What I run now

MAI-Transcribe-2 in verbatim mode transcribes the whole recording in one request, at ten cents an hour. pyannote separates the speakers locally and, because it returns a voice embedding per speaker, the app can match them against voices I have named before, so a recurring colleague comes back as a name instead of "Speaker 2". A language-model cleanup pass then tidies the verbatim text with the changes visible. Whisper stays as the free, offline option, and after this benchmark I know exactly what it costs me on meetings.

Gemini, with the 30-minute cap, is now a real alternative on single-speaker audio, and I would not have known that without finding my own bug. Scribe is the other one I would reach for, on a recording where the names matter more than the token cost.

## Caveats

This is five recordings. The ranking held on all five, but the margins are not to be quoted to a decimal, and the top two are a tenth of a point apart on the earnings calls. The accents are US, UK and the AMI participants' non-native English. The proper-noun metric is a heuristic. The speaker metrics depend on the references' own segmentation, which for AMI includes overlapping speech no engine here transcribes twice. And I built the pipeline every engine ran through, so where that pipeline is at fault, as it was for Gemini, the engine wears it until someone checks. That is an argument for publishing the tooling, which is in the repository under `tools/`, along with the full issue log and the process log.

## If you are choosing

If you have a GPU and patience, and your recordings are mostly one voice at a time: Whisper large-v3 with pyannote and a vocabulary you let grow. It costs nothing and nothing leaves the building. On meetings with overlap it will lose a fifth of the words, and the turbo model is no worse than the full one.

If you want the least setup and can live with the quota: ElevenLabs Scribe. Second to MAI on every recording by a point or less, best at names on every recording, the only cloud engine that tells you how sure it is, and back in seconds. The reasons not to are the token cost and that your audio leaves the building.

If you are already in Google's ecosystem: Gemini in verbatim mode, and never send it more than 30 minutes at a time no matter what it accepts. On single-speaker audio it is within a point of the best.

If you want the most accurate transcription of a long meeting for the least money: MAI-Transcribe-2 in verbatim mode for the words, with speaker separation done on your side. Do not turn on its diarization for anything longer than a few minutes, and do not use its clean style.

And if you are paying a monthly fee to transcribe your own recordings, you probably don't have to. The code is all there: https://github.com/labatt/transcriber-studio

#speechrecognition #transcription #opensource #whisper #azure #gemini #python
