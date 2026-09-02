---
name: run-transcriber-studio
description: Build, run, and drive Transcriber Studio, the PySide6 desktop app. Use when asked to start or launch the app, take a screenshot of its UI, click something in it, reproduce a UI bug, check a widget's state, or run its tests.
---

Transcriber Studio is a **PySide6 (Qt) desktop app**. Drive it with
`.claude/skills/run-transcriber-studio/driver.py` — a stdin REPL that launches
`MainWindow` **in-process** and gives you the real widget objects. Prefer it
over clicking screen pixels: it can read any widget's text, click buttons by
label, and screenshot via `QWidget.grab()`, which renders offscreen and so
works even when the window is covered or minimised.

Paths below are relative to the repo root. Verified on Windows 11, Python
3.13, PySide6 6.11.2.

## Prerequisites

No system packages needed on Windows — a working Python with the project's
deps installed is enough. Verify:

```bash
python -c "import PySide6; print(PySide6.__version__)"   # 6.11.2
```

**Always export this first.** The app's log and account label contain `…`, `✓`
and `—`; without it they arrive as `?` or crash the pipe on Windows' cp1252
console:

```bash
export PYTHONIOENCODING=utf-8
```

## Run (agent path) — the driver

One command per line on stdin. Every command answers with a line starting
`OK ` or `ERR `, so you can block until you see one.

```bash
export PYTHONIOENCODING=utf-8
printf 'ready\nstate\ntabs\nss 01-startup\ntab Local\nss 02-local-files\nquit\n' \
  | python .claude/skills/run-transcriber-studio/driver.py
```

Screenshots land in `.claude/skills/run-transcriber-studio/shots/<name>.png`.
**Open them and look** — a blank or tofu-filled frame means something is wrong.

### Commands

| Command | Does |
|---|---|
| `ready` | banner once the window is up |
| `wait <ms>` | pump the event loop (nested loop, so app timers keep running) |
| `waitfor <expr> [ms]` | poll until a Python expression is truthy (default 10000ms) |
| `ss [name] [attr]` | screenshot window, or one widget attribute |
| `tree [filter]` | dump widget tree: class / objectName / text / visible |
| `tabs`, `tab <index\|title>` | list / switch tabs |
| `click <label\|objectName>` | click a button: exact, then *unique prefix*. Never a bare substring. |
| `click! <label>` | override the guard on a destructive button (see Gotchas) |
| `text <attr>`, `log`, `state` | read a widget, the log pane, or a summary line |
| `eval <expr>` / `exec <stmt>` | arbitrary Python; `w` is MainWindow, `app` the QApplication |
| `dismiss` | close any open modal dialog |
| `quit` | close and exit 0 |

`eval` is the escape hatch — reach anything the table doesn't cover:

```bash
export PYTHONIOENCODING=utf-8
printf 'waitfor w.recordings_tab.table.rowCount() > 0 60000\nclick Select all\neval sum(1 for r in range(w.recordings_tab.table.rowCount()) if w.recordings_tab.table.item(r,0).checkState().value==2)\nss 03-recordings\nquit\n' \
  | python .claude/skills/run-transcriber-studio/driver.py --real-config
```

That run printed `OK eval 43` — 43 recordings loaded from the live Plaud API
and all checked. It needs a signed-in Plaud session; when logged out the table
stays empty, the `waitfor` times out, and `state` reads
`account='Not logged in to Plaud'`. Log in via the `Login` button in the app
(`python run.py`) — it opens a browser flow that cannot be driven from here.

### Config modes — pick deliberately

| Flag | Config | Use for |
|---|---|---|
| *(default)* | temp dir, seeded `setup_complete: true` | most work. Cannot touch real settings. |
| `--real-config` | the real `settings.json` | reproducing bugs that need real settings or the restored job queue |
| `--fresh-config` | temp dir, `setup_complete: false` | exercising the first-run `SetupWizard` |
| `--offscreen` | (combines with the above) | headless/CI — **but see Gotchas**, text renders as boxes |

`--real-config` is safe for read-only inspection: `MainWindow` has no
`closeEvent`, and `config.save()` is only reached from explicit user actions
(the wizard, the Settings dialog). Don't click `Setup`/`Settings` under it.

## Run (human path)

```bash
python run.py
```

Opens the window and blocks; Ctrl-C to stop. Fine for a human, useless to an
agent — there is no handle on the widgets.

## Test

```bash
export PYTHONIOENCODING=utf-8
python -m pytest -q      # 392 passed in 49.87s
```

## Gotchas

- **Startup is ~6s, and it is all `import torch`.** `MainWindow.__init__`
  calls `_update_gpu_badge()` (`transcriber_studio/ui/main_window.py:160`) →
  `cuda_available()` (`transcriber_studio/hardware.py:54`) → `import torch`,
  synchronously, before `show()`. Measured: 5.77s of a 6.5s warm start; the
  app's own imports are 0.52s. First launch of the day is far worse (cold
  DLLs, Defender, OneDrive-backed checkout) — allow 60s before concluding it
  hung. Budget `waitfor` timeouts accordingly.
- **`--offscreen` renders every glyph as a tofu box and drops the dark
  stylesheet.** Layout, checkboxes and geometry are still correct, so it is
  fine for structural assertions — but never read text off an offscreen
  screenshot. Read text with `eval`/`text`/`state` instead. Also note offscreen
  grabs are 1x (1330x820) while the onscreen ones are 2x DPI (2360x1640).
- **Isolating the config does NOT log you out.** Plaud tokens live in
  `~/.plaud/tokens.json` (see `transcriber_studio/plaud_web.py:9`), outside
  `APP_DIR`. Even the default temp-dir mode shows the real signed-in account
  and can hit the live API.
- **Patch `CONFIG_PATH`, not just `APP_DIR`.** Both are module-level in
  `config.py` and read at call time by `load()`/`save()`, so patching `APP_DIR`
  alone still writes the user's real `settings.json`. `isolate_config()` in the
  driver sets both; `tests/support.py` does the same for its stores.
- **The first-run `SetupWizard` is modal** when `setup_complete` is false.
  It does *not* deadlock the driver — verified that the command timer still
  ticks inside the wizard's nested `exec()` loop — but every command until you
  `dismiss` acts on a window with a dialog on top of it.
- **Commands run faster than the app's async work.** A bare `state` right after
  launch reports `account='Checking Plaud login…'` and an empty table. Gate on
  the thing you actually need, e.g.
  `waitfor w.recordings_tab.table.rowCount() > 0 60000`.
- **`click` refuses Logout / Clear all / Remove selected.** Their effects
  outlive the process. This guard exists because an earlier substring matcher
  turned `click Go` into a click on **Logout** ('go' is inside 'Logout'), which
  deleted `~/.plaud/tokens.json` and cost an interactive browser re-login.
  Matching is now exact-then-unique-prefix, and these need `click! <label>`.
- **Don't `time.sleep()` inside a command** — it freezes the Qt loop and the
  app makes no progress. `wait` uses a nested `QEventLoop` for this reason.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `account='Checking Plaud login?'`, mangled `—`/`✓` | `export PYTHONIOENCODING=utf-8` |
| Nothing for ~60s after launch, no window | Normal cold start (torch). Wait before killing it. |
| `ERR waitfor timed out after 10000ms` | Default budget is 10s and startup alone is 6s — pass an explicit budget: `waitfor <expr> 60000` |
| Screenshot is all `□` boxes | You are on `--offscreen`. Drop it, or read text via `eval` instead. |
| `ERR RuntimeError: button '▶  Go' is disabled` | Real app state — `Go` is disabled until rows are selected. `click Select all` first. |
| `ERR RuntimeError: 'Logout' has effects that outlive the process` | The destructive-button guard. Use `click! Logout` only if you mean it. |
| `ERR KeyError: "no button matching 'X'; visible: [...]"` | The error lists every visible label — pick one from it. |
| `ERR KeyError: 'X' is ambiguous: [...]` | Prefix hit several buttons; use the full label. |
