# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin wrapper around the official `plaud` CLI.

Authentication is handled entirely by the CLI (browser OAuth, tokens stored at
~/.plaud/tokens.json). We never touch credentials directly — we only shell out
to the locally installed `plaud` binary and parse its human-readable output.

Tested against plaud CLI 0.2.4.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .job_cancel import check_cancel
from .models import Recording, Source

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HEX32_RE = re.compile(r"^([0-9a-f]{32})\s+(.*?)\s{2,}(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$")
KV_RE = re.compile(r"^\s*([a-z_]+):\s*(.*)$")
URL_RE = re.compile(r"https?://\S+")

# On Windows the CLI is `plaud.cmd`; resolve whatever is on PATH.
PLAUD_BIN = shutil.which("plaud") or "plaud"

# Fixed OAuth callback port used by the official `plaud` CLI (cannot be changed).
PLAUD_CALLBACK_PORT = 8199

# Node/libuv fast-fail on Windows after the CLI has already printed valid output.
_WINDOWS_CLI_CRASH_EXIT = -1073740791  # NTSTATUS 0xC0000409


class PlaudError(RuntimeError):
    pass


class NotAuthenticated(PlaudError):
    pass


@dataclass
class Account:
    id: str
    email: str
    nickname: str


def _strip(text: str) -> str:
    return ANSI_RE.sub("", text)


def _sanitize_cli_output(text: str) -> str:
    """Drop libuv crash noise that the Plaud CLI sometimes prints on Windows."""
    lines = []
    for line in text.splitlines():
        if "UV_HANDLE_CLOSING" in line or line.strip().startswith("Assertion failed"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_benign_windows_cli_crash(proc: subprocess.CompletedProcess[str]) -> bool:
    """True when `plaud` printed output then aborted during Node shutdown on Windows."""
    if proc.returncode == 0:
        return False
    rc = proc.returncode
    if rc == _WINDOWS_CLI_CRASH_EXIT or rc == _WINDOWS_CLI_CRASH_EXIT + 2**32:
        return True
    blob = (proc.stdout or "") + (proc.stderr or "")
    return "UV_HANDLE_CLOSING" in blob


def _port_bind_error(port: int, err: OSError) -> str:
    """Human-readable guidance when the OAuth callback port cannot be bound."""
    win_err = getattr(err, "winerror", None)
    errno = getattr(err, "errno", None)
    in_use = errno in (98, 10048) or win_err == 10048
    blocked = errno in (13, 10013) or win_err == 10013

    if in_use:
        return (
            f"Port {port} is already in use. Close any other Plaud login window "
            f"or `plaud-mcp` process, then try again."
        )

    if blocked and sys.platform == "win32":
        return (
            f"Windows is blocking port {port}, which Plaud login requires.\n\n"
            "This usually happens when Docker Desktop, WSL2, or Hyper-V reserves "
            "a large port range that includes 8199.\n\n"
            "Fix (PowerShell as Administrator):\n"
            "  1. Close Docker Desktop / stop WSL if possible\n"
            "  2. net stop winnat\n"
            f"  3. netsh int ipv4 add excludedportrange protocol=tcp startport={port} "
            f"numberofports=1 store=persistent\n"
            "  4. net start winnat\n"
            "  5. Retry Login (reboot if the port is still blocked)\n\n"
            "To inspect reserved ranges:\n"
            "  netsh interface ipv4 show excludedportrange protocol=tcp"
        )

    return f"Could not bind OAuth callback port {port}: {err}"


def probe_callback_port(port: int = PLAUD_CALLBACK_PORT) -> None:
    """Raise PlaudError if the Plaud CLI cannot listen for OAuth callbacks."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        raise PlaudError(_port_bind_error(port, e)) from e
    finally:
        sock.close()


def format_login_error(message: str) -> str:
    """Turn raw `plaud login` stderr into a clearer message for the UI."""
    lower = message.lower()
    if "callback port" in lower or "port_probe_failed" in lower:
        if "eacces" in lower or "10013" in lower or "permission denied" in lower:
            return _port_bind_error(
                PLAUD_CALLBACK_PORT,
                OSError(13, "permission denied"),
            )
        if "eaddrinuse" in lower or "already in use" in lower:
            return _port_bind_error(
                PLAUD_CALLBACK_PORT,
                OSError(10048, "address already in use"),
            )
    return message.strip() or "Plaud login failed."


def _parse_duration(s: str) -> float:
    """'1h09m' / '16m59s' / '3s' -> seconds (best effort)."""
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", s):
        v = float(value)
        total += v * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


# Plaud's API answers 500 often enough that a single one should not end a job:
# the same recording usually resolves seconds later. Anything matching these is
# retried; a plain "not available" is not, because that answer will not change
# until the device finishes uploading.
TRANSIENT_MARKERS = (
    "500", "502", "503", "504", "internal server error", "bad gateway",
    "service unavailable", "gateway time-out", "fetch_failed", "timed out",
    "timeout", "etimedout", "econnreset", "socket hang up", "enotfound", "eai_again",
)
AUDIO_URL_ATTEMPTS = 3


def _is_transient(message: str) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in TRANSIENT_MARKERS)


class PlaudClient:
    def __init__(self, bin_path: str = PLAUD_BIN, timeout: int = 60):
        self.bin = bin_path
        self.timeout = timeout

    # ---- low level ---------------------------------------------------------
    def _run(self, args: list[str], timeout: int | None = None) -> str:
        try:
            proc = subprocess.run(
                [self.bin, *args],
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                shell=False,
            )
        except FileNotFoundError as e:
            raise PlaudError(
                "The 'plaud' CLI was not found on PATH. Install it with "
                "`npm install -g @plaud-ai/cli` (requires Node.js >= 20)."
            ) from e
        except subprocess.TimeoutExpired as e:
            # Otherwise this escapes as a bare TimeoutExpired and reaches the
            # jobs table as an unreadable repr.
            raise PlaudError(
                f"The plaud CLI did not answer within {timeout or self.timeout}s "
                f"(command: plaud {' '.join(args)}). Plaud may be busy — try again."
            ) from e
        out = _sanitize_cli_output(
            _strip((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
        )
        if proc.returncode == 2:
            raise NotAuthenticated("Not logged in to Plaud. Use the Login button.")
        if proc.returncode != 0:
            if _is_benign_windows_cli_crash(proc) and out.strip():
                return out
            raise PlaudError(out.strip() or f"plaud exited with code {proc.returncode}")
        return out

    # ---- auth --------------------------------------------------------------
    def is_cli_installed(self) -> bool:
        return shutil.which(self.bin) is not None or self.bin == PLAUD_BIN

    def me(self) -> Account | None:
        try:
            out = self._run(["me"])
        except NotAuthenticated:
            return None
        fields: dict[str, str] = {}
        for line in out.splitlines():
            m = KV_RE.match(line)
            if m:
                fields[m.group(1)] = m.group(2).strip()
        if not fields.get("id"):
            return None
        return Account(
            id=fields.get("id", ""),
            email=fields.get("email", ""),
            nickname=fields.get("nickname", ""),
        )

    def login(self) -> None:
        # `plaud login` opens a browser; allow generous timeout for the flow.
        probe_callback_port()
        try:
            self._run(["login"], timeout=300)
        except PlaudError as e:
            raise PlaudError(format_login_error(str(e))) from e

    def logout(self) -> None:
        self._run(["logout"])

    # ---- listing -----------------------------------------------------------
    def list_files(self, page: int = 1, page_size: int = 50) -> list[Recording]:
        page_size = max(10, min(100, page_size))
        out = self._run(["files", "--page", str(page), "--page-size", str(page_size)])
        return self._parse_file_table(out)

    def recent(self, days: int = 7) -> list[Recording]:
        out = self._run(["recent", "--days", str(days)])
        return self._parse_file_table(out)

    def search(self, keyword: str, max_results: int = 100) -> list[Recording]:
        out = self._run(["search", keyword])
        recs = self._parse_file_table(out)
        return recs[:max_results]

    def _parse_file_table(self, out: str) -> list[Recording]:
        recs: list[Recording] = []
        for line in out.splitlines():
            m = HEX32_RE.match(line.strip())
            if not m:
                continue
            rid, name, date, duration = m.groups()
            name = name.replace("…", "").strip()  # drop truncation ellipsis
            recs.append(
                Recording(
                    source=Source.PLAUD,
                    id=rid,
                    name=name,
                    date=date,
                    duration=duration,
                    duration_seconds=_parse_duration(duration),
                )
            )
        return recs

    # ---- detail / audio ----------------------------------------------------
    def get_file(self, file_id: str) -> Recording:
        out = self._run(["file", file_id])
        f: dict[str, str] = {}
        for line in out.splitlines():
            m = KV_RE.match(line)
            if m:
                f[m.group(1)] = m.group(2).strip()
        return Recording(
            source=Source.PLAUD,
            id=f.get("id", file_id),
            name=f.get("name", file_id),
            date=(f.get("start_at", "") or f.get("created_at", ""))[:10],
            datetime=f.get("start_at", "") or f.get("created_at", ""),
            duration=f.get("duration", ""),
            duration_seconds=_parse_duration(f.get("duration", "")),
            audio_available=f.get("audio", "").lower() == "available",
            serial_number=f.get("serial_number", ""),
        )

    def audio_listed_available(self, file_id: str) -> bool:
        """What the file's own metadata claims about its audio."""
        try:
            return self.get_file(file_id).audio_available
        except PlaudError:
            return False

    def audio_url(self, file_id: str, log_cb=None) -> str | None:
        """The signed 24h URL, or None when Plaud really has no audio for this file.

        Neither kind of failure here is reliable on its own. Plaud's API returns
        the occasional 500, and — measured against four recordings that the
        phone app, the web UI and `plaud file` all agreed were in the cloud —
        it also answers a flat "Audio not available" for audio it lists as
        available, then hands over a URL minutes later. So a refusal is only
        believed when the file's own metadata agrees the audio is not there;
        otherwise it is treated as the hiccup it usually is, and retried.
        """
        last_error = ""
        for attempt in range(1, AUDIO_URL_ATTEMPTS + 1):
            try:
                out = self._run(["audio", file_id])
            except NotAuthenticated:
                raise
            except PlaudError as e:
                last_error = str(e)
                if not _is_transient(last_error) or attempt == AUDIO_URL_ATTEMPTS:
                    raise
            else:
                for line in out.splitlines():
                    m = URL_RE.search(line)
                    if m:
                        return m.group(0)
                refused = "not available" in out.lower()
                if refused and not self.audio_listed_available(file_id):
                    return None         # believed: the cloud copy is not there
                last_error = (
                    'Plaud answered "not available" for audio it lists as available'
                    if refused else "Plaud returned no audio URL"
                )
                if attempt == AUDIO_URL_ATTEMPTS:
                    return None
            delay = 2 ** attempt
            if log_cb:
                first_line = last_error.strip().splitlines()[0] if last_error.strip() else "failed"
                log_cb(
                    f"Plaud audio URL: {first_line} — retrying in {delay}s "
                    f"({attempt}/{AUDIO_URL_ATTEMPTS})."
                )
            time.sleep(delay)
        return None

    def download_audio(
        self, file_id: str, dest_path: str, progress_cb=None, should_cancel=None,
        label: str = "", log_cb=None,
    ) -> str:
        """Resolve the 24h URL and stream the mp3 to dest_path. Returns dest_path.

        Streams to a .part file and only moves it into place once complete, so
        a cancelled or crashed download never leaves a truncated file that a
        later run would mistake for cached audio.

        The .part file is kept, not deleted, and the next attempt asks the
        server to carry on from where it stopped. An hour-long recording
        interrupted at ninety percent used to start again from nothing, which
        is a long way to fall for a laptop closing its lid.
        """
        url = self.audio_url(file_id, log_cb=log_cb)
        if not url:
            name = label or file_id
            if self.audio_listed_available(file_id):
                # The cloud copy exists; Plaud just would not hand over a URL.
                raise PlaudError(
                    f"Plaud lists audio for “{name}” but would not return a download "
                    f"link after {AUDIO_URL_ATTEMPTS} attempts. That is a Plaud-side "
                    "hiccup, not a missing recording — wait a few minutes and run it "
                    "again."
                )
            raise PlaudError(
                f"Plaud has no cloud audio for “{name}” yet.\n"
                "The recording uploads from the device after it finishes — open the "
                "Plaud app to let it sync, then press Refresh and run it again."
            )
        partial = Path(f"{dest_path}.part")
        already = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={already}-"} if already else {}
        if already and log_cb:
            log_cb(f"Resuming download at {already / 1e6:.1f} MB.")
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                # 206 means it honoured the range; anything else means it is
                # sending the file from the top and the old bytes are useless.
                resumed = r.status_code == 206 and already > 0
                if not resumed and already:
                    if log_cb:
                        log_cb("The server would not resume — starting the download again.")
                    already = 0
                total = int(r.headers.get("Content-Length", 0)) + already
                done = already
                with open(partial, "ab" if resumed else "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        check_cancel(should_cancel, message="Download cancelled.")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if progress_cb and total:
                            progress_cb(done / total)
            partial.replace(dest_path)
        except BaseException:
            # Deliberately left on disk: it is the head start for next time,
            # and .part is never mistaken for the finished file.
            raise
        return dest_path
