# Four speech-to-text engines and one Plaud recorder

I own a Plaud recorder. It's a good little device, and the recordings it makes are the kind most transcription tools handle worst: a table mic in a meeting room, someone talking two tables over, a phone call on speaker. Plaud will transcribe those for you, for a monthly fee. I got tired of paying to transcribe audio I'd already made on hardware I'd already bought, so I built my own transcription app. While building it I wired in four different speech engines and worked with each of them.

The app is Transcriber Studio: https://github.com/labatt/transcriber-studio. It's open source under the GPL, it runs on your own machine, it imports directly from a Plaud account or takes any audio file, and it lets you pick the engine per run. That last part is what made this comparison possible. The same recordings go through the same pipeline, and only the decoder changes.

I have not run formal accuracy benchmarks. Everything below is what I hit building the integrations and using them on my own recordings. Where I measured something, I say so and give the number. Where I'm describing a service's documented behaviour, I say that instead.

## Why the model is only part of the result

The design bet behind the app is that the model is maybe a third of the result. The other two thirds are what happens to the audio before the model hears it, and what the model is told to expect. A speech model decoding noisy audio leans on its language prior, and that is where a surname turns into a common noun that sounds like it. So the app carries a vocabulary of names and jargon forward from one recording to the next, weighted by how often each term has been seen, and hands it to whichever engine can take hints. It runs voice activity detection so silence never reaches the decoder. It can denoise first. And it separates speakers as a distinct step, which turned out to matter more than I expected. The Microsoft section explains why.

## Whisper, running locally

This is OpenAI's Whisper via faster-whisper, on your own GPU or CPU.

Pros. It's free, and the audio never leaves the machine. Large-v3 is the default on a GPU, and it's the model I'd still pick for hard audio with a lot of proper nouns. CrisperWhisper, a verbatim fine-tune, keeps every filler, stutter and false start that stock Whisper drops. That matters if you want the words as spoken and plan to clean them up afterwards where you can see what changed. faster-whisper's hotwords mechanism re-injects your vocabulary into every decoder window, so a name learned in January still biases the decode in September. The app budgets that list at 600 characters by default, under the roughly 850 that fit in half the prompt window before the tail gets silently truncated.

It also composes with local speaker separation in a way nothing else here does. pyannote (speaker-diarization-community-1) runs after the decode, and because it returns a voice embedding per speaker, the app can match those against voices you've enrolled before. Name a speaker once, tick "remember this voice", and later recordings can come back with a real name instead of "Speaker 2". None of the cloud engines return the voice data that makes that possible.

Cons. The cost is time. Large-v3 needs about 10 GB of VRAM in float16, an hour of audio is many minutes of GPU work, and on CPU it runs slower than the recording itself. CrisperWhisper is slower still, roughly realtime even on a GPU, because a verbatim model writes down every disfluency and Whisper decodes one token at a time. Large-v3-turbo is about six times faster with a small accuracy cost, but it gets there by cutting the decoder from 32 layers to four. The decoder is where hotwords act, so vocabulary biasing has less to work with. Speaker separation needs a HuggingFace token and accepting the terms on three gated models, which is a real setup hurdle. And Whisper's failure mode on bad audio is inventing fluent text. The app turns off carry-over context between windows and watches for speech-free stretches inside "speech" segments to limit that, but it doesn't eliminate it.

If you go this route, know that faster-whisper's batched inference pipeline is a real speedup, but its source hardcodes the hallucination-silence threshold to off and ignores what you pass. I only found that by reading the installed code. "Free and local" costs more in attention than it looks.

## ElevenLabs Scribe

Pros. It's the easy button. One call transcribes and separates speakers, up to 32 of them, and returns word-level timings. It accepts files up to 5 GB, so I never had to think about splitting. It can tag non-speech events like laughter and applause if you want them. Language comes back as a code, the integration is a single multipart upload, and there's little to configure.

Cons. There's no voice embedding in the response, so speakers are numbered per recording and can't be matched across recordings. Every transcript is "Speaker 1, Speaker 2" until you rename them. Your audio leaves your machine and is billed to your ElevenLabs account. And because it does everything in one call, there's no seam where you can intervene: no chance to feed it a vocabulary the way Whisper takes hotwords, and no separate diarization step to tune. What comes back is what you get.

I haven't benchmarked Scribe's accuracy against the others on the same files, so I won't claim a ranking. It is the least work and the least control of the four.

## Gemini 3.5 Transcribe

Pros. Google's transcription model is reachable with the same API key the app already uses for LLM cleanup, so there's nothing extra to sign up for if you're already using Gemini. In "verbatim" mode it returns speakers and word timings, and the raw transcription quality on my recordings has been fine.

Cons. The API is awkward for long recordings. There are two modes, and only one is usable for this job. "Smart" returns a nicely written block of prose with no speakers and no timestamps, because the API refuses both parameters in that mode. Google documents a 60-minute limit per request, or 30 minutes with speaker separation, so an hour-long meeting has to be cut into parts. The app cuts at silence and overlaps each part by three minutes so the same speech is heard twice, which is what lets it match "Speaker 2" in part one to "Speaker 1" in part two. That works, but it's machinery I had to build and a place where things can go wrong. Gemini also won't accept custom vocabulary alongside speakers or timestamps, so it gets none of the glossary the other engines benefit from.

One related finding is about Gemini as an LLM rather than as a transcriber. When I used a Gemini "flash" model for the cleanup pass, it was spending its output budget thinking before it answered. On one real batch it burned 15,728 tokens of a 16,384 budget on reasoning and truncated the actual JSON. The fix was to turn thinking off for a task that is mechanical restructuring, but until I measured it, the failure looked like batches that were too big.

## MAI-Transcribe-2

Microsoft's new model, reached through Azure Speech's fast transcription endpoint in "enhanced mode". It's in public preview. This is where I spent the most interesting week.

Pros. Transcription is fast and cheap. A 73-minute, 70 MB MP3 went up in a single request, untouched, and the transcript came back seconds after the upload finished, at a promotional $0.10 per hour of audio through the end of 2026. It returns word-level timings. It takes a phrase list as recognition hints, so the app's accumulated vocabulary carries over, and it offers a choice between "verbatim" and "clean" output styles.

Cons, measured. First, the phrase list has a hard cap of 50 entries that the documentation doesn't mention. A 64-term glossary made every request fail with a 400 until I bisected it. Second, "clean" style is lossy beyond fillers. On my test clip it removed a whole real clause, dropping the word count from 74 to 61, so I run verbatim and let the cleanup pass do the tidying where I can see it. Third, the response includes a confidence field per phrase, and it's zero on every phrase. The field exists in the schema but the model doesn't fill it.

The big one is speaker diarization. With diarization enabled, a 25-second clip worked. Fifteen minutes returned HTTP 408 "Timeout", three times. Thirty minutes returned HTTP 503 wrapping a diarization error. Seventy-three minutes returned 408, then 500, then 503. The exact same files, with diarization turned off, transcribed perfectly, the 73-minute one included. I also re-encoded the 73-minute file down to 17.5 MB and it still failed with diarization on, so file size has nothing to do with it. The failure is in MAI's diarization component.

What makes this a con rather than a footnote is how it presents. The errors read like network problems, and the documentation gives you no reason to suspect otherwise. The quotas page says 5 hours, the REST reference says 2, the MAI page says 300 MB and no duration, and nothing anywhere mentions a diarization limit. I lost real time on the wrong theory before isolating it. I've opened a documentation pull request and a Q&A thread with the full table of runs.

There's also no way around it inside Microsoft's services. The batch transcription API accepts blob URLs and handles hours of audio, but it drops the enhanced-mode parameter without telling you and runs a different model. Passing a URL to the fast endpoint is refused the moment a MAI model is named. So the audio has to go up in the request.

## What I ended up with

MAI does two things well, transcribing fast and taking hints, and one thing badly, separating speakers on long audio. That split lines up with a seam the app already had. So MAI now runs as a decoder only. It transcribes the whole recording in one request, and pyannote separates the speakers locally afterwards, exactly as it does for Whisper. That sidesteps the diarization failure entirely, and because pyannote returns voice embeddings, voiceprints work on this engine too. Then an LLM cleanup pass tidies the verbatim text, with the changes visible.

The result is fast and cheap, the speaker data stays on my machine, and there's no subscription. If you want zero cost and zero data leaving the building, Whisper alone still does the job. It just takes longer.

## What I'd tell someone choosing

- If you have a GPU and patience, and you care about names and privacy: Whisper large-v3 with pyannote and a vocabulary you let grow.
- If you want the least setup and don't need speakers matched across recordings: ElevenLabs Scribe.
- If you're already in Google's ecosystem and your recordings are under half an hour: Gemini, in verbatim mode only.
- If you want speed and low cost and are willing to run speaker separation yourself: MAI-Transcribe-2 for the words, pyannote for the speakers. Don't turn on its diarization for anything longer than a few minutes, and use verbatim.

If you're paying a monthly fee to transcribe your own recordings, you probably don't have to. The code is all there.

#speechrecognition #transcription #opensource #whisper #azure #python
