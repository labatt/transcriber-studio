I got tired of paying Plaud a monthly fee to transcribe recordings I'd already made on a Plaud device. So I built my own transcription app, and along the way wired it to four different speech engines and worked with each of them. This is what I found. There are no formal benchmarks here, only what I hit building and using it.

The app is Transcriber Studio: https://github.com/labatt/transcriber-studio. Open source (GPL), runs on your own machine. It pulls recordings straight from a Plaud account or takes any audio file, and lets you pick the engine per run. The design bet behind it is that the model is maybe a third of the result. The rest is what happens to the audio before the model hears it, and what the model is told to expect.

The four engines:

WHISPER (local, faster-whisper). It's free and the audio never leaves the machine. Large-v3 on a GPU, with CrisperWhisper as the option that keeps every filler and false start if you want true verbatim. The cost is time: an hour of audio is many minutes of GPU work, and on CPU it runs slower than the recording itself. Speaker separation is a second step with pyannote, but that second step is also what makes voiceprints possible. Enrol a voice once and later transcripts can come back with a real name instead of "Speaker 2". None of the three cloud engines below return the voice data that makes that work.

ELEVENLABS SCRIBE. Transcribes and separates speakers in one call, returns word-level timings, accepts files up to 5 GB, and can tag non-speech events like laughter and applause. It's the least setup of the four. What you give up is that your audio leaves your machine and there's no voice data to match speakers across recordings.

GEMINI 3.5 TRANSCRIBE. Two modes. "Smart" returns a block of written prose with no speakers and no timestamps, because the API refuses both in that mode. "Verbatim" gives you speakers and timings. Google documents a 60-minute limit per request, or 30 with speaker separation, so long recordings get split and the speakers matched back up across the seams. It also won't take custom vocabulary alongside speakers or timestamps.

MAI-TRANSCRIBE-2 (Microsoft, via Azure Speech, in public preview). Transcription is fast and cheap: a 73-minute, 70 MB file went up in one request and came back seconds after the upload finished, at a promotional $0.10 per hour of audio through the end of 2026. It takes your glossary as recognition hints, with a hard cap of 50 terms that the documentation doesn't mention. It offers "clean" and "verbatim" styles. On my test clip, "clean" removed a whole real clause, not just filler words, so I run verbatim.

The catch is speaker diarization. A 25-second clip worked. Fifteen minutes returned HTTP 408 "Timeout" three times. Thirty minutes returned 503 with a diarization error. Seventy-three minutes returned 408, then 500, then 503. The exact same files transcribed perfectly with diarization turned off. The errors read like a network problem, but they aren't one. Microsoft's quotas page says 5 hours, the REST reference says 2, and nothing documents a diarization limit at all. I've opened a docs PR and a Q&A thread with the evidence.

So the setup I've landed on: MAI transcribes the whole recording in one shot, pyannote separates the speakers locally, voiceprints can put names on them, and an LLM cleanup pass tidies the verbatim text where I can see what it changed. It's fast and cheap, the speaker data stays on my machine, and there's no subscription. Whisper alone costs nothing at all.

If you're paying a monthly fee to transcribe your own recordings, you probably don't have to. The code is all there.

#speechrecognition #transcription #opensource #whisper #azure #python
