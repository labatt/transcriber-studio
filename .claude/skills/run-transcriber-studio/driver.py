#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Agent-facing REPL driver for Transcriber Studio.

Launches MainWindow in-process and reads one command per line on stdin, so an
agent can drive the real Qt widgets instead of guessing at screen pixels.
Screenshots use QWidget.grab(), which renders offscreen -- it works when the
window is covered, minimised, or on the offscreen Qt platform.

    python .claude/skills/run-transcriber-studio/driver.py [--real-config] [--offscreen]

Commands (one per line, on stdin):
    ready               print a banner once the window is up
    wait <ms>           pump the event loop for <ms> (async work needs this)
    waitfor <expr> [ms] poll until a Python expression is truthy (default 10000ms)
    ss [name]           screenshot the window -> shots/<name>.png
    ss <name> <attr>    screenshot one widget, e.g. `ss justtabs tabs`
    tree [filter]       dump the widget tree (class / objectName / text)
    tabs                list tab titles and which is current
    tab <index|title>   switch tabs
    click <label|objName>  click a button; exact then unique-prefix match
    click! <label>      same, but allow a button in DESTRUCTIVE (e.g. Logout)
    text <attr>         print the text of a widget attribute on the window
    log                 print the app's on-screen log pane
    state               one-line summary: account, gpu badge, tab, queue size
    eval <expr>         eval a Python expression; `w` is MainWindow, `app` the QApplication
    exec <stmt>         exec a Python statement (same names in scope)
    dismiss             close any open modal dialog (see Gotchas in SKILL.md)
    quit                close the window and exit 0

Every command prints a line starting with `OK ` or `ERR `, so a caller can
block until it sees one.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from transcriber_studio.suppress_warnings import configure as _configure_warnings

_configure_warnings()

from transcriber_studio.hardware import configure_cuda_dll_paths

configure_cuda_dll_paths()

# Buttons whose effects escape the app and outlive the process. `click` refuses
# these; `click!` is the deliberate override, so a loose label never trips one.
DESTRUCTIVE = {"logout", "clear all", "remove selected"}


def isolate_config(tmpdir: Path, fresh: bool = False) -> None:
    """Point the app at a throwaway config dir seeded as already set up.

    Two reasons this is not optional by default:
      * APP_DIR and CONFIG_PATH are read at call time by config.load/save, so
        patching only APP_DIR still writes the user's real settings.json.
      * A config with setup_complete=false pops a MODAL SetupWizard on the
        first event-loop turn. Verified: the driver's QTimer still ticks inside
        the wizard's nested exec() loop, so `dismiss` clears it -- but every
        command until then acts on a window with a dialog on top of it.
        Pass --fresh-config to get that state deliberately.
    """
    from transcriber_studio import config

    tmpdir.mkdir(parents=True, exist_ok=True)
    config.APP_DIR = tmpdir
    config.CONFIG_PATH = tmpdir / "settings.json"
    config.CONFIG_PATH.write_text(
        json.dumps({"setup_complete": not fresh}), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real-config",
        action="store_true",
        help="use the real settings/Plaud login instead of a temp dir "
        "(may pop the modal setup wizard; will write to the real settings.json)",
    )
    ap.add_argument(
        "--fresh-config",
        action="store_true",
        help="isolated config seeded as NOT set up, to exercise the first-run SetupWizard",
    )
    ap.add_argument("--offscreen", action="store_true", help="force the offscreen Qt platform")
    ap.add_argument("--shots", default=str(HERE / "shots"), help="screenshot output directory")
    args = ap.parse_args()

    if args.offscreen:
        import os

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    tmp = None
    if not args.real_config:
        tmp = tempfile.TemporaryDirectory(prefix="ts-driver-")
        isolate_config(Path(tmp.name), fresh=args.fresh_config)

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QAbstractButton, QApplication, QDialog, QWidget

    from transcriber_studio import components, config
    from transcriber_studio.ui.main_window import MainWindow

    components.refresh_path()

    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    # ~6s: MainWindow.__init__ -> _update_gpu_badge -> cuda_available -> import torch
    w = MainWindow()
    w.show()

    cmds: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in sys.stdin:
            cmds.put(line.rstrip("\n"))
        cmds.put("quit")

    threading.Thread(target=pump, daemon=True).start()

    def out(msg: str) -> None:
        print(msg, flush=True)

    def norm(s: str) -> str:
        """Button label -> comparable text: drop the & accelerator and any
        decoration, so '▶  Go' matches 'Go'."""
        return " ".join(s.replace("&", "").split()).strip(" ▶✓….—-").strip()

    def find_button(needle: str) -> QAbstractButton:
        """Exact first, then a UNIQUE prefix. Never a bare substring.

        A substring pass once made `click Go` match 'Logout' -- 'go' is inside
        'Logout' -- and clicking Logout deleted the real ~/.plaud/tokens.json.
        Destructive buttons sit next to innocuous ones here, so an ambiguous
        needle must raise, not guess.
        """
        buttons = w.findChildren(QAbstractButton)
        for b in buttons:
            if b.objectName() == needle:
                return b
        want = norm(needle).lower()
        exact = [b for b in buttons if norm(b.text()).lower() == want]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise KeyError(f"{needle!r} matches {len(exact)} buttons; use an objectName")
        pre = [b for b in buttons if norm(b.text()).lower().startswith(want)]
        if len(pre) == 1:
            return pre[0]
        if len(pre) > 1:
            raise KeyError(f"{needle!r} is ambiguous: {sorted(norm(b.text()) for b in pre)}")
        raise KeyError(
            f"no button matching {needle!r}; visible: "
            f"{sorted(norm(b.text()) for b in buttons if b.isVisible() and norm(b.text()))}"
        )

    def grab(name: str, attr: str | None) -> str:
        target = getattr(w, attr) if attr else w
        path = shots / f"{name}.png"
        pix = target.grab()
        if not pix.save(str(path)):
            raise RuntimeError(f"QPixmap.save failed for {path}")
        return f"{path}  {pix.width()}x{pix.height()}"

    def handle(line: str) -> bool:
        """Return False to stop the loop."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()

        if cmd == "quit":
            out("OK quit")
            return False
        if cmd == "wait":
            # A nested event loop, so timers/threads the app is waiting on keep
            # running. A bare time.sleep() here would freeze them instead.
            ms = int(rest or 1000)
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()
            out(f"OK wait {ms}")
        elif cmd == "waitfor":
            # The trailing token is only a timeout if the whole line does NOT
            # parse on its own -- otherwise `waitfor w.x.count() > 999` loses
            # its 999 to the budget and evaluates the syntax error `... >`.
            expr, budget = rest, "10000"
            try:
                compile(rest, "<waitfor>", "eval")
            except SyntaxError:
                head, _, tail = rest.rpartition(" ")
                if tail.isdigit() and head:
                    expr, budget = head, tail
            deadline = time.monotonic() + int(budget) / 1000
            while time.monotonic() < deadline:
                if eval(expr, {"w": w, "app": app}):
                    out(f"OK waitfor {expr}")
                    return True
                loop = QEventLoop()
                QTimer.singleShot(100, loop.quit)
                loop.exec()
            out(f"ERR waitfor timed out after {budget}ms: {expr}")
        elif cmd == "ready":
            out(f"OK ready title={w.windowTitle()!r} visible={w.isVisible()}")
        elif cmd == "ss":
            parts = rest.split()
            name = parts[0] if parts else "shot"
            attr = parts[1] if len(parts) > 1 else None
            out(f"OK ss {grab(name, attr)}")
        elif cmd == "tree":
            n = 0
            for c in w.findChildren(QWidget):
                label = c.__class__.__name__
                if rest and rest.lower() not in label.lower() and rest.lower() not in c.objectName().lower():
                    continue
                getter = getattr(c, "text", None)
                txt = str(getter()) if callable(getter) else ""
                out(f"   {label:<22} name={c.objectName()!r:<16} text={txt[:40]!r} vis={c.isVisible()}")
                n += 1
            out(f"OK tree {n} widgets")
        elif cmd == "tabs":
            titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
            out(f"OK tabs current={w.tabs.currentIndex()} {titles}")
        elif cmd == "tab":
            if rest.isdigit():
                w.tabs.setCurrentIndex(int(rest))
            else:
                for i in range(w.tabs.count()):
                    if rest.lower() in w.tabs.tabText(i).lower():
                        w.tabs.setCurrentIndex(i)
                        break
                else:
                    raise KeyError(f"no tab matching {rest!r}")
            out(f"OK tab {w.tabs.currentIndex()} {w.tabs.tabText(w.tabs.currentIndex())!r}")
        elif cmd in ("click", "click!"):
            b = find_button(rest)
            if norm(b.text()).lower() in DESTRUCTIVE and cmd != "click!":
                raise RuntimeError(
                    f"{norm(b.text())!r} has effects that outlive the process "
                    f"(Logout deletes ~/.plaud/tokens.json and needs an "
                    f"interactive browser login to undo). Use `click! {rest}` "
                    f"if you really mean it."
                )
            if not b.isEnabled():
                raise RuntimeError(f"button {b.text()!r} is disabled")
            b.click()
            out(f"OK click {b.text()!r}")
        elif cmd == "text":
            out(f"OK text {getattr(w, rest).text()!r}")
        elif cmd == "log":
            out(w.log.toPlainText())
            out("OK log")
        elif cmd == "state":
            out(
                f"OK state account={w.account_label.text()!r} gpu={w.gpu_badge.text()!r} "
                f"tab={w.tabs.tabText(w.tabs.currentIndex())!r} queue={len(w._queue_recordings)}"
            )
        # eval/exec are the point of a debugging REPL: they let an agent reach
        # any widget or method without the driver growing a command per field.
        # The only input is the stdin of a locally launched process -- the same
        # trust boundary as `python -i`. Never wire this to a socket or a file
        # anything else can write.
        elif cmd == "eval":
            out(f"OK eval {eval(rest, {'w': w, 'app': app})!r}")
        elif cmd == "exec":
            exec(rest, {"w": w, "app": app})
            out("OK exec")
        elif cmd == "dismiss":
            n = 0
            for d in app.topLevelWidgets():
                if isinstance(d, QDialog) and d.isVisible():
                    d.reject()
                    n += 1
            out(f"OK dismiss {n}")
        else:
            out(f"ERR unknown command {cmd!r}")
        return True

    def tick() -> None:
        try:
            line = cmds.get_nowait()
        except queue.Empty:
            return
        try:
            if not handle(line):
                timer.stop()
                w.close()
                app.quit()
        except Exception as exc:
            out(f"ERR {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(50)

    out("OK driver up -- MainWindow shown, awaiting commands")
    rc = app.exec()
    if tmp is not None:
        tmp.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
