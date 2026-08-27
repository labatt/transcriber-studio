# Transcriber Studio

A desktop workbench that turns recordings into clean, speaker-labelled transcripts — on your own machine.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/labatt/transcriber-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/labatt/transcriber-studio/actions/workflows/tests.yml)

Most transcription tools give you one knob: which model. On hard audio — a noisy room, a table
mic, someone talking two tables over — the model is maybe a third of the result. This app is built
around the other two thirds: **what happens to the audio before the decoder sees it**, and **what
the decoder is told to expect**.

> **Not affiliated with, endorsed by, or sponsored by PLAUD AI.** PLAUD is a trademark of its
> respective owner. This is an independent tool that can import from PLAUD cloud recorders using
> their public CLI, alongside ordinary local audio files.

---

## What it does

**A three-layer front end, then the model.**

1. **Denoise** — [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) cleans the signal first.
   Whisper decodes noisy speech by leaning harder on its language prior, which is exactly when it
   starts inventing fluent text that was never said. Noise suppression is the cheapest accuracy in
   the pipeline.
2. **Voice activity detection** — Silero VAD (bundled with faster-whisper) cuts silence and
   non-speech so the decoder never sees it. This removes Whisper's worst failure mode at the
   source: hallucinating a paragraph over a quiet stretch.
3. **Vocabulary biasing** — names, product names and jargon are fed to the decoder as hotwords
   before it starts guessing at them. The words come from your **shared glossary**, which fills
   itself in as jobs run: what one recording taught the app, the next one already knows.

Then Whisper (or ElevenLabs Scribe) for the words, [pyannote](https://github.com/pyannote/pyannote-audio)
for who said them, and an optional LLM pass to turn fragments into readable prose.

**Also in the box**

- **Shared glossaries** — named vocabularies several jobs read from and write back to. Import,
  combine and dedupe them; where two sources disagree (same term, different type), the entry is
  tagged for review instead of one reading being picked silently.
- **AI cleanup** — merge fragments into sentences, fix speaker attribution, normalise terms
  against the glossary, and flag anything garbled. Works with OpenRouter, OpenAI, Anthropic,
  Google, xAI, and Ollama (cloud or local).
- **Resume** — an interrupted job restarts from its last saved step. No re-transcription, no
  repeated model calls, no repeated spend.
- **Components window** — what's installed, what's newer, and buttons to update it, including the
  CUDA-aware PyTorch upgrade that a plain `pip install --upgrade torch` gets wrong.
- **Output formats** — `txt`, `srt`, `vtt`, `json`, `md`, with a filename template builder.

---

## Requirements

|                    | Minimum                                  | Recommended                                   |
| ------------------ | ---------------------------------------- | --------------------------------------------- |
| OS                 | Windows 10/11                            | Windows 11                                     |
| Python             | 3.10                                     | 3.12 or 3.13                                   |
| RAM                | 8 GB                                     | 16 GB+                                         |
| Disk               | ~3 GB (small model + deps)               | ~15 GB (large-v3 + CUDA PyTorch)               |
| GPU                | none — CPU works                         | NVIDIA, 8 GB+ VRAM, driver 525+                |
| Also needed        | [ffmpeg](https://ffmpeg.org) on PATH     | + `deep-filter` binary, Node 20+ for PLAUD     |

**Windows is the tested platform.** The app runs on Linux (Qt, ffmpeg and the ML stack are all
cross-platform) but the installer helpers assume Windows: the Components window shells out to
`winget`, and the data directory follows `%APPDATA%`. macOS is untested; there is no CUDA there,
so it would be the CPU or cloud path. Patches welcome.

Nothing here needs a GPU. Without one you can still run local Whisper on CPU, or send audio to
ElevenLabs Scribe and skip local compute entirely — see below.

---

## GPU vs. no GPU

This is the decision that shapes everything else, so here it is plainly.

### With an NVIDIA GPU

Whisper `large-v3` runs several times faster than real time, and speaker detection runs on the GPU
too. This is the configuration the app is tuned for.

- **VRAM** decides which model fits: `large-v3` wants ~10 GB in float16, `medium` ~5 GB, `small`
  ~2 GB. With 8 GB you can usually still run `large-v3` if nothing else is on the card; the app
  falls back to CPU automatically if the GPU load fails.
- **PyTorch must be a CUDA build.** `pip install torch` gives you the CPU one, and everything will
  appear to work while speaker detection crawls. Install it from the CUDA index:

  ```
  pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu126
  ```

  Pick the channel that matches your driver (`nvidia-smi` reports the highest CUDA it supports).
  **CUDA 12 is deliberately preferred over CUDA 13** here, because CTranslate2 — the engine that
  actually runs Whisper — is built against CUDA 12 and cuDNN 9. The Components window works this
  out for you and refuses to pin a channel that no longer publishes the release you're moving to.
- Whisper and diarization are separate stacks. Whisper runs on CTranslate2, diarization on PyTorch;
  the app reports each one's device independently in Settings, because it is entirely possible for
  one to be on the GPU and the other not.

### Without a GPU

Everything still works. Three honest options:

1. **Local Whisper on CPU** — use `small` (the app recommends it automatically). Expect roughly
   real time to a few times slower: an hour of audio in one to three hours. `large-v3` on CPU is
   possible but usually not worth the wait.
2. **ElevenLabs Scribe** — a cloud engine that transcribes *and* diarizes in one pass, with no
   local compute and no HuggingFace token. Costs money, and the audio leaves your machine.
3. **Skip diarization** — it is the most expensive part on CPU. If you only need the words, turn
   speaker detection off.

The denoise and VAD layers are cheap on CPU either way — DeepFilterNet processes several minutes of
audio per second on a laptop CPU, and it matters more on bad audio than the model size does. If you
only take one thing from this README: **on hard audio, denoise + VAD + biasing on `small` beats
`large-v3` on the raw file.**

---

## Install

### From source (Windows)

```powershell
git clone https://github.com/labatt/transcriber-studio.git
cd transcriber-studio

python -m venv .venv
.venv\Scripts\Activate.ps1

# Core app + the local Whisper engine
pip install -e ".[local]"

# Optional: local speaker diarization (install CUDA torch FIRST if you have a GPU)
pip install -e ".[diarization]"

python run.py
```

An installed copy also gets a `transcriber-studio` command.

### External tools

| Tool | Needed for | Install |
| --- | --- | --- |
| **ffmpeg** | decoding, channel splitting, the fallback denoiser — **required** | `winget install Gyan.FFmpeg` |
| **deep-filter** | DeepFilterNet denoising (strongly recommended) | [download the release binary](https://github.com/Rikorose/DeepFilterNet/releases/latest), then point Settings → Audio front-end at it |
| **Node 20+ and the PLAUD CLI** | importing from a PLAUD cloud account | `winget install OpenJS.NodeJS.LTS` then `npm install -g @plaud-ai/cli` |

> **Why a binary for DeepFilterNet?** The `deepfilternet` pip package depends on `deepfilterlib`,
> which has no wheels past Python 3.11 and fails to build from source on newer Pythons. The
> standalone `deep-filter` executable is the same model and does not care which Python you run.
> With neither installed the app falls back to ffmpeg's own `afftdn` denoiser — real, but clearly
> weaker on background chatter, and the UI says so rather than pretending otherwise.

The app's **Components** window (Settings → Components & updates) checks all of these, reports
versions against the current releases, and can install or update them for you.

### First run

A setup wizard covers the engine, the model, and the service keys. Two things it can't do for you:

- **Speaker detection** needs a free HuggingFace token *and* you must click "Agree and access" on
  three gated model pages ([speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
  [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
  [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)). The
  wizard's Test button verifies all three.
- **Your name.** Output files are named after the *other* person on a recording, so Options →
  Output → "Your name(s)" is how the app knows which speaker is you. List every spelling you get
  labelled with (`Alex Rivera, Alex R, Alex`). Leave it empty and the first named speaker is used.

---

## Models

| Model | Parameters | VRAM (fp16) | Speed | Accuracy |
| --- | --- | --- | --- | --- |
| `tiny` | 39M | ~1 GB | ~10× faster | Roughest |
| `base` | 74M | ~1 GB | ~7× faster | Rough |
| `small` | 244M | ~2 GB | ~4× faster | Decent |
| `medium` | 769M | ~5 GB | ~2× faster | Good |
| `large-v2` | 1550M | ~10 GB | baseline | Excellent |
| `large-v3` | 1550M | ~10 GB | baseline | Best |
| **CrisperWhisper** | 1550M | ~10 GB | about the same | Best (verbatim) |

Speed is relative to `large-v3` on the same machine. Models download from HuggingFace on first use.

**[CrisperWhisper](https://huggingface.co/nyralabs/CrisperWhisper)** is a `large-v3` fine-tune that
transcribes *verbatim* — it keeps the fillers, stutters and false starts stock Whisper quietly tidies
away, and it hallucinates less over noise. Pair it with AI Cleanup, which is where tidying belongs:
after the words are on the page. Three caveats, all shown in the app:

- **English and German only.** The fine-tune was trained on those two.
- **Word timestamps are less precise** than the original — this is the CTranslate2 conversion,
  which computes them differently. Segment times are unaffected.
- ⚠️ **The weights are CC-BY-NC-4.0 — non-commercial.** Every other model here is MIT. If you are
  transcribing for commercial purposes, use `large-v3` instead. See [Licensing](#licensing).

---

## Where your data lives

Everything is local, in `%APPDATA%\TranscriberStudio`:

```
settings.json      your settings and API keys, in plain text
queue.json         the job queue, so it survives a restart
history.json       what was processed and when
glossaries/        the shared glossary library
audio_cache/       downloaded PLAUD audio
denoise_cache/     enhanced audio, so a re-run does not redo it
resume/            checkpoints for interrupted jobs
```

Transcripts go wherever you point the output folder — never into the app directory.

**What leaves your machine:** nothing, unless you turn it on. Local Whisper, DeepFilterNet, VAD and
pyannote all run on your hardware. Audio is uploaded only if you choose the ElevenLabs engine.
Transcript text is sent to an LLM provider only if you enable AI Cleanup, and only to the provider
you picked. API keys are stored locally and sent only to the service they belong to.

---

## Troubleshooting

**"ffmpeg is not installed" right after I installed it.** Windows hands each process a copy of the
environment at launch, and installers only edit the registry — so a running app can't see a new
PATH. Worse, winget installs ffmpeg into a version-stamped folder, so an *upgrade* deletes the
directory the app started with. The app now re-reads PATH from the registry on startup, after any
install, and on every component scan; if it still can't find it, restart the app.

**Speaker detection is very slow.** PyTorch is probably the CPU build. Settings reports
"Diarization: CPU" when this is the case; reinstall torch from the CUDA index (above).

**Whisper hallucinated a paragraph that was never said.** Turn on VAD and the hallucination guard
in Options → Audio pipeline, and denoise the input. That combination exists specifically for this.

**A transcript keeps mangling the same name.** Put it in a shared glossary and point the job at it —
the next run biases the decoder toward it. Or type it into Options → Audio pipeline → Extra
vocabulary for a one-off.

**GPU load failed / cuBLAS or cuDNN errors.** The app falls back to CPU and says so in the log.
Usually a CUDA/PyTorch mismatch — check the Components window.

---

## Development

```powershell
pip install -e ".[local,dev]"
pytest                 # 184 tests, no GPU or network needed
ruff check .
```

Tests run headless (`QT_QPA_PLATFORM=offscreen`) and never touch your real settings, glossaries or
caches — see `tests/support.py` for the isolation helpers. If you add state that lives in
`APP_DIR`, add an isolation helper for it too.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Licensing

Transcriber Studio is **GPL-3.0-or-later**. See [LICENSE](LICENSE).

Third-party components keep their own licences, and two are worth knowing about:

| Component | Licence | Note |
| --- | --- | --- |
| PySide6 / Qt | LGPL-3.0 | Fine for this source release. If you ever ship a **bundled binary**, LGPL obligations apply — ship the Qt libraries as separate shared libraries and include their licence. |
| CrisperWhisper weights | CC-BY-NC-4.0 | **Non-commercial only.** An optional model, not a dependency. |
| faster-whisper, CTranslate2, pyannote.audio, onnxruntime | MIT | |
| PyTorch | BSD-3-Clause | |
| DeepFilterNet | MIT / Apache-2.0 | |
| OpenAI Whisper weights | MIT | |
| pyannote models | gated | Free, but you must accept the terms on HuggingFace. |

This is a summary offered in good faith, not legal advice. If you plan to redistribute or use this
commercially, read the licences yourself.
