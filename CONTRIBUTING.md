# Contributing

Thanks for looking. This is a small project with opinions; here they are, so a pull request does
not run into them by surprise.

## Getting set up

```powershell
git clone https://github.com/labatt/transcriber-studio.git
cd transcriber-studio
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[local,dev]"
pytest
```

The tests need no GPU, no network and no API keys. If a change makes that untrue, the change is
probably in the wrong place.

## What the code values

- **Degrade, do not fail.** A missing denoiser, an unreachable index, a model that will not load on
  the GPU — none of these should end a job. Fall back, say so in the log, carry on.
- **Say what actually happened.** "Silero VAD v6 (bundled with faster-whisper 1.2.1)" beats "VAD
  enabled". If the app fell back to a weaker path, the UI should admit it rather than imply the
  good one ran. Never report a state you did not check: "not checked yet" and "checked, no answer"
  are different facts.
- **Comments explain why, not what.** The interesting comments here are the ones recording a trap:
  why `-D` is not optional, why torchcodec is in the torch upgrade set, why PATH is re-read from
  the registry. If you fix something subtle, leave the reason behind.
- **Tests describe behaviour.** `test_an_explicitly_chosen_backend_is_never_silently_swapped`, not
  `test_resolve_2`. A test name should survive a refactor of the thing it tests.

## Before opening a pull request

- `pytest` passes.
- `ruff check .` is clean.
- New state under `APP_DIR` has an isolation helper in `tests/support.py`, and the tests use it.
  Tests that write into a real settings, glossary or cache directory will be sent back — they
  corrupt the machine of whoever runs them.
- No personal data in code, comments, tests or fixtures. Use invented names.
- New files carry the SPDX header:

  ```python
  # SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```

  Contributions are accepted under GPL-3.0-or-later. Keep your own copyright line if you prefer.

## Things worth doing

- **macOS support.** Untested. The Components window assumes `winget`, and the data directory
  assumes `%APPDATA%`.
- **Batched inference.** faster-whisper ships `BatchedInferencePipeline`; the app does not use it
  yet, and it is the obvious speedup.
- **A denoiser that is not Windows-first.** The `deep-filter` binary exists for Linux and macOS;
  the install help only explains the Windows one.
- **Timestamp drift verification.** Denoiser alignment currently rests on `deep-filter -D` doing
  what it claims, not on a measurement. A test that proves it would be genuinely useful.

## Reporting bugs

Include the job log — the app logs which denoiser, which VAD, which model and which device it
actually used, which is usually the whole answer. Redact anything from a real recording first.
