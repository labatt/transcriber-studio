# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the app is running on, what is newer, and how to update it.

Every piece here is something the app depends on but cannot ship: a pip
package, a CLI on PATH, a downloaded binary. Each one knows three things —
which version is installed, which is current, and the command that closes the
gap — so the UI can present them uniformly and the answers can be tested
without a dialog on screen.

Nothing in this module changes the machine. Building an update command and
running it are deliberately separate: the running command is the user's
decision, made in front of the exact text that will execute.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from . import audio_utils, config, denoise, diarization

#: How the versions we find compare.
UP_TO_DATE = "current"
OUTDATED = "outdated"
MISSING = "missing"
UNCHECKED = "unchecked"    # installed; the index has not been asked yet
NOT_NEEDED = "not_needed"  # absent, but something that does the same job is here
UNKNOWN = "unknown"        # installed; the index was asked and did not answer

PIP = [sys.executable, "-m", "pip"]
NETWORK_TIMEOUT = 8

_VERSION_RE = re.compile(r"(\d+(?:\.\d+){1,3})")


@dataclass(frozen=True)
class Component:
    key: str
    title: str
    why: str
    #: "pip" | "npm" | "winget" | "download" — decides how updates happen.
    kind: str
    package: str = ""
    url: str = ""
    optional: bool = True
    #: Set when an update needs the app restarted to take effect.
    restart_required: bool = False
    #: Another component that does the same job. When that one is installed,
    #: this one is not missing — it is simply not needed.
    satisfied_by: str = ""
    install_steps: list[str] = field(default_factory=list)
    notes: str = ""


COMPONENTS: list[Component] = [
    Component(
        key="ffmpeg",
        title="ffmpeg",
        why="Decodes the audio, splits channels, and runs the fallback denoiser.",
        kind="system",
        package="ffmpeg",
        url="https://ffmpeg.org/download.html",
        optional=False,
        install_steps=["Install it, then restart this app so it picks up the new PATH:"],
    ),
    Component(
        key="plaud",
        title="Plaud CLI",
        why="Lists and downloads your Plaud cloud recordings.",
        kind="npm",
        package="@plaud-ai/cli",
        url="https://nodejs.org/en/download",
        install_steps=["Needs Node.js 20 or newer, then:"],
    ),
    Component(
        key="faster-whisper",
        title="faster-whisper",
        why="The local transcription engine, and the Silero VAD that feeds it.",
        kind="pip",
        package="faster-whisper",
        restart_required=True,
        notes=(
            "Updating this can also move the bundled Silero VAD to a newer "
            "version, which is usually what you want."
        ),
    ),
    Component(
        key="ctranslate2",
        title="CTranslate2",
        why="Runs the Whisper weights. faster-whisper pins what it needs.",
        kind="pip",
        package="ctranslate2",
        restart_required=True,
        notes="Let faster-whisper choose this one unless you have a reason not to.",
    ),
    Component(
        key="pyannote.audio",
        title="pyannote.audio",
        why="Local speaker diarization — who said which line.",
        kind="pip",
        package="pyannote.audio",
        restart_required=True,
    ),
    Component(
        key="torch",
        title="PyTorch",
        why="Runs diarization, and the DeepFilterNet Python backend if you use it.",
        kind="pip",
        package="torch",
        restart_required=True,
        notes=(
            "On a CUDA build the update deliberately reinstalls torch, "
            "torchvision and torchaudio together from the CUDA index — a plain "
            "pip upgrade would pull the CPU wheel and take the GPU away from "
            "speaker detection."
        ),
    ),
    Component(
        key="deep-filter",
        title="DeepFilterNet (deep-filter binary)",
        why=(
            "Noise suppression in front of the decoder — worth more on hard audio "
            "than a bigger Whisper model."
        ),
        kind="download",
        url="https://github.com/Rikorose/DeepFilterNet/releases/latest",
        install_steps=[
            "Download the build for this machine from the releases page "
            "(look for the asset whose name matches your platform).",
            "Point Settings → Audio front-end at it, or put it on your PATH as "
            "deep-filter.",
        ],
    ),
    Component(
        key="deepfilternet",
        title="DeepFilterNet (Python package)",
        why="The same denoiser as a pip package, for Pythons it has wheels for.",
        kind="pip",
        package="deepfilternet",
        restart_required=True,
        satisfied_by="deep-filter",
        notes=(
            "Its wheels lag new Python releases; on a Python it does not support, "
            "install the deep-filter binary above instead."
        ),
    ),
    Component(
        key="PySide6",
        title="PySide6 (this app's UI)",
        why="The window you are looking at.",
        kind="pip",
        package="PySide6",
        restart_required=True,
        notes="Updating the toolkit under a running app needs a restart to take effect.",
    ),
]

BY_KEY = {c.key: c for c in COMPONENTS}


# ---- keeping PATH current ---------------------------------------------
def refresh_path() -> list[str]:
    """Pick up PATH entries added since this process started. Returns the new ones.

    Windows hands every process a copy of the environment at launch, and an
    installer that edits PATH only edits the registry. The effect is an app that
    reports a dependency as missing minutes after it was installed — worse,
    winget installs ffmpeg into a version-stamped folder, so an *upgrade* leaves
    this process pointing at a directory that no longer exists.

    Entries are added, never removed: an activated virtualenv puts its own
    directory on PATH and nowhere else, and replacing PATH wholesale from the
    registry would take it away.
    """
    if sys.platform != "win32":
        return []
    import winreg

    sources = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    stored: list[str] = []
    for root, key_path in sources:
        try:
            with winreg.OpenKey(root, key_path) as key:
                value, _kind = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        stored.extend(part for part in os.path.expandvars(value).split(os.pathsep) if part)

    current = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    have = {part.rstrip("\\").lower() for part in current}
    added = [part for part in stored if part.rstrip("\\").lower() not in have]
    if added:
        os.environ["PATH"] = os.pathsep.join(current + added)
    return added


# ---- what is installed ------------------------------------------------
def executable(name: str) -> str:
    """Resolve a CLI to a full path before running it.

    npm and the Plaud CLI are .CMD shims on Windows, and CreateProcess — which
    is what subprocess uses without a shell — will not find those by bare name
    the way a shell does. Resolving first is the difference between "not
    installed" and the version it actually reports.
    """
    return shutil.which(name) or name


def _run_version(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, errors="replace"
        )
    except Exception:
        return ""
    text = f"{proc.stdout}\n{proc.stderr}"
    match = _VERSION_RE.search(text)
    return match.group(1) if match else ""


def _pip_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return ""


def ffmpeg_version() -> str:
    if not audio_utils.have_ffmpeg():
        return ""
    return _run_version([audio_utils.FFMPEG, "-version"])


def plaud_version() -> str:
    found = shutil.which("plaud")
    if found is None:
        return ""
    return _run_version([found, "version"])


def deep_filter_version(configured: str = "") -> str:
    binary = denoise.binary_path(configured)
    if not binary:
        return ""
    # Older builds answer -V; if that fails the binary is still there and
    # usable, so report the fact rather than nothing.
    return _run_version([binary, "-V"]) or "installed"


def installed_version(component: Component, settings=None) -> str:
    """The version on this machine, or "" when the component is absent."""
    if component.key == "ffmpeg":
        return ffmpeg_version()
    if component.key == "plaud":
        return plaud_version()
    if component.key == "deep-filter":
        path = (settings.deep_filter_path if settings else config.load().deep_filter_path)
        return deep_filter_version(path)
    if component.key == "pyannote.audio" and not diarization.is_available():
        return ""
    return _pip_version(component.package)


def torch_cuda_build() -> str:
    """The CUDA version PyTorch was built against, "" for a CPU build."""
    try:
        import torch

        return str(torch.version.cuda or "")
    except Exception:
        return ""


# ---- what is current --------------------------------------------------
def latest_version(component: Component) -> str:
    """Ask the component's own index what the current release is.

    Never raises: an update check that cannot reach the network reports
    "unknown", which is materially different from "up to date" and is shown
    that way.
    """
    try:
        if component.kind == "pip":
            return _json_field(
                f"https://pypi.org/pypi/{component.package}/json", ("info", "version")
            )
        if component.kind == "npm":
            return _json_field(
                f"https://registry.npmjs.org/{component.package}/latest", ("version",)
            )
        if component.key == "deep-filter":
            tag = _json_field(
                "https://api.github.com/repos/Rikorose/DeepFilterNet/releases/latest",
                ("tag_name",),
            )
            return tag.lstrip("v")
        if component.key == "ffmpeg":
            return _text("https://www.gyan.dev/ffmpeg/builds/release-version")
    except Exception:
        return ""
    return ""


def _json_field(url: str, path: tuple[str, ...]) -> str:
    import requests

    data = requests.get(url, timeout=NETWORK_TIMEOUT).json()
    for key in path:
        data = data[key]
    return str(data or "")


def _text(url: str) -> str:
    import requests

    body = requests.get(url, timeout=NETWORK_TIMEOUT).text.strip()
    match = _VERSION_RE.search(body)
    return match.group(1) if match else ""


def parse_version(value: str) -> tuple:
    """Comparable form of a version string; unparseable input sorts lowest.

    Only the numeric release matters here. A local suffix like "+cu124" says
    which build of the same release is installed, not that it is newer or
    older, so it is left out of the comparison and shown separately.
    """
    match = _VERSION_RE.search(value or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def compare(installed: str, latest: str) -> str:
    if not installed:
        return MISSING
    if not latest:
        return UNKNOWN
    left, right = parse_version(installed), parse_version(latest)
    if not left or not right:
        return UNKNOWN
    # Pad so 1.2 and 1.2.0 compare equal rather than by length.
    width = max(len(left), len(right))
    left += (0,) * (width - len(left))
    right += (0,) * (width - len(right))
    return OUTDATED if left < right else UP_TO_DATE


# ---- rights ------------------------------------------------------------
def is_elevated() -> bool:
    """True when this process is already running as administrator."""
    if sys.platform != "win32":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


@functools.lru_cache(maxsize=1)
def site_packages_writable() -> bool:
    """Can pip write where this interpreter keeps its packages?

    Deliberately does not test by creating a file: writing a probe into
    site-packages can sit for a minute behind a real-time virus scanner, and
    this is called while a dialog is being drawn. os.access is instant, and
    where it is wrong on Windows it errs toward "yes" — so the command runs, and
    the failure it produces is caught and offered a retry, rather than the UI
    silently choosing a scope nobody asked for.
    """
    import sysconfig

    target = sysconfig.get_paths().get("purelib", "")
    if not target or not os.path.isdir(target):
        return False
    return os.access(target, os.W_OK)


def pip_scope() -> str:
    """How a pip install has to be run here: "normal", "user", or "elevated".

    Installing into the user's own site-packages is the right answer to a
    read-only system Python — it needs no rights at all, and it is what pip
    itself suggests. It is invalid inside a virtualenv, where the environment
    is writable anyway.
    """
    if site_packages_writable():
        return "normal"
    if in_virtualenv():
        return "elevated"       # a venv nobody can write to needs rights, not --user
    return "user"


def needs_elevation(component: Component) -> bool:
    """Whether this component's command is likely to be refused without rights."""
    if is_elevated():
        return False
    if component.kind == "pip":
        return pip_scope() == "elevated"
    if component.kind == "system":
        # apt needs root; winget and brew normally do not. We cannot know a
        # winget package's scope in advance, so the flags keep it from hanging
        # and the output says if it was refused.
        return _platform_key() == "linux"
    return False


def can_elevate() -> bool:
    """Whether this app can re-run a command with more rights on its own.

    Only on Windows, where the UAC flow is a documented, tested path. There is
    no equivalent here for sudo/pkexec/osascript that has been exercised, and a
    privilege-escalation path nobody has run is not something to ship — the UI
    shows the command to run by hand instead.
    """
    return sys.platform == "win32"


#: Fragments Windows and pip use when the answer is "you may not write there".
PERMISSION_MARKERS = (
    "access is denied",
    "permission denied",
    "winerror 5",
    "consider using the `--user` option",
    "requires elevation",
    "administrator",
    "not writeable",
    "not writable",
)


def looks_like_a_permission_problem(output: str) -> bool:
    lowered = (output or "").lower()
    return any(marker in lowered for marker in PERMISSION_MARKERS)


def elevated_command(cmd: list[str], log_path: str) -> list[str]:
    """Re-launch a command as administrator, capturing its output to a file.

    The elevated process is a child of the UAC consent flow, not of this app, so
    its output cannot be piped back the ordinary way — it is redirected to a
    file the caller reads once the wait returns.
    """
    def ps_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    arguments = ", ".join(ps_quote(part) for part in cmd[1:])
    script = (
        f"Start-Process -FilePath {ps_quote(cmd[0])} "
        + (f"-ArgumentList {arguments} " if len(cmd) > 1 else "")
        + f"-Verb RunAs -Wait -WindowStyle Hidden "
        f"-RedirectStandardOutput {ps_quote(log_path)} "
        f"-RedirectStandardError {ps_quote(log_path + '.err')}"
    )
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]


# ---- what would change it ---------------------------------------------
#: winget will otherwise stop and wait for a keypress this app cannot deliver.
WINGET_FLAGS = [
    "--accept-source-agreements",
    "--accept-package-agreements",
    "--disable-interactivity",
]

#: How each platform installs a system package, for components of kind
#: "system". The value is (manager, install args, update args); a platform with
#: no entry gets no command and is pointed at the project's own download page.
SYSTEM_PACKAGES: dict[str, dict[str, tuple[str, list[str], list[str]]]] = {
    "ffmpeg": {
        "win32": ("winget", ["install", "--id", "Gyan.FFmpeg", *WINGET_FLAGS],
                  ["upgrade", "--id", "Gyan.FFmpeg", *WINGET_FLAGS]),
        "darwin": ("brew", ["install", "ffmpeg"], ["upgrade", "ffmpeg"]),
        "linux": ("apt-get", ["install", "-y", "ffmpeg"], ["install", "-y", "--only-upgrade", "ffmpeg"]),
    },
}


def _platform_key() -> str:
    """sys.platform, collapsed to the three families the tables key on."""
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def system_command(component: Component, *, update: bool) -> list[str]:
    """The package-manager command for this component on this platform.

    Empty when the platform has no entry — the caller falls back to pointing at
    the download page, which is better than printing a command that does not
    exist here.
    """
    table = SYSTEM_PACKAGES.get(component.package if component.kind == "system" else "", {})
    entry = table.get(_platform_key())
    if not entry:
        return []
    manager, install_args, update_args = entry
    return [executable(manager), *(update_args if update else install_args)]


def _pip_install(args: list[str], component: Component, version: str = "") -> list[str]:
    """Assemble a pip command in the order a person would read it."""
    cmd = _torch_aware([*PIP, "install", *args], component, version)
    if pip_scope() == "user" and not is_elevated():
        # The system's own site-packages is read-only for this account; the
        # user's is not, and needs no rights at all.
        cmd.append("--user")
    return cmd


def install_command(component: Component, version: str = "") -> list[str]:
    """The command that installs the component for the first time."""
    if component.kind == "pip":
        return _pip_install([component.package], component, version)
    if component.kind == "npm":
        return [executable("npm"), "install", "-g", component.package]
    if component.kind == "system":
        return system_command(component, update=False)
    return []


def update_command(component: Component, version: str = "") -> list[str]:
    """The command that moves the component to the current release.

    ``version`` is what the index says is current; for torch it decides which
    CUDA channel can actually supply it.
    """
    if component.kind == "pip":
        return _pip_install(["--upgrade", component.package], component, version)
    if component.kind == "npm":
        return [executable("npm"), "install", "-g", f"{component.package}@latest"]
    if component.kind == "system":
        return system_command(component, update=True)
    return []       # a downloaded binary is replaced by hand


#: PyTorch's CUDA channels, newest first within each major. CUDA 12 comes
#: first on purpose: CTranslate2 — the engine that actually runs Whisper — is
#: built against CUDA 12 and cuDNN 9, and the app puts those pip-installed
#: NVIDIA libraries on the DLL search path. Moving torch to a CUDA 13 channel
#: is not wrong, but it is a bigger change than an update button should make on
#: its own.
CUDA_CHANNELS = ("cu130", "cu129", "cu128", "cu126", "cu124", "cu121")
CUDA_12_CHANNELS = tuple(c for c in CUDA_CHANNELS if c.startswith("cu12"))


def installed_cuda_channel() -> str:
    """The channel this torch came from, e.g. "cu124"; "" for a CPU build."""
    cuda = torch_cuda_build()
    return "cu" + cuda.replace(".", "")[:3] if cuda else ""


@functools.lru_cache(maxsize=32)
def channel_has(channel: str, version: str) -> bool:
    """Does this CUDA channel publish `version` for this Python and platform?

    The installed channel is not a safe default for an upgrade: PyTorch retires
    a channel and simply stops publishing to it, so pinning cu124 to fetch a
    release that only exists on cu126 finds nothing, reports success, and
    changes nothing at all.
    """
    import sysconfig

    import requests

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    try:
        body = requests.get(
            f"https://download.pytorch.org/whl/{channel}/torch/", timeout=NETWORK_TIMEOUT
        ).text
    except Exception:
        return False
    needle = f"torch-{version}%2B{channel}-{python_tag}-{python_tag}-{platform_tag}.whl"
    return needle in body


def torch_channel_for(version: str) -> str:
    """The CUDA channel to install `version` from, or "" to use PyPI's CPU wheel.

    Prefers the channel already in use, then the rest of the CUDA 12 line, and
    only then CUDA 13 — see CUDA_CHANNELS.
    """
    if not torch_cuda_build():
        return ""
    installed = installed_cuda_channel()
    order = [installed] if installed else []
    order += [c for c in CUDA_12_CHANNELS if c != installed]
    order += [c for c in CUDA_CHANNELS if c not in order]
    for channel in order:
        if channel_has(channel, version):
            return channel
    return installed


def _torch_aware(cmd: list[str], component: Component, version: str = "") -> list[str]:
    """Keep a CUDA PyTorch a CUDA PyTorch, from a channel that actually has it.

    Plain `pip install --upgrade torch` pulls the CPU wheel from PyPI and
    silently takes the GPU away from diarization, and torch/torchvision/
    torchaudio have to move as a set.
    """
    if component.key != "torch":
        return cmd
    # torchcodec belongs to the set too: pyannote decodes audio through it, and
    # it is built against a specific torch ABI, so leaving it behind turns a
    # torch upgrade into a diarization crash.
    cmd = cmd + ["torchvision", "torchaudio", "torchcodec"]
    if not torch_cuda_build():
        return cmd
    channel = torch_channel_for(version) if version else installed_cuda_channel()
    if not channel:
        return cmd
    return cmd + ["--index-url", f"https://download.pytorch.org/whl/{channel}"]


def command_text(cmd: list[str]) -> str:
    """The command as a person would type it, quoting only what needs it."""
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


@dataclass(frozen=True)
class Status:
    component: Component
    installed: str
    latest: str
    state: str
    detail: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.state in (MISSING, OUTDATED)

    def summary(self) -> str:
        if self.state == MISSING:
            return "not installed"
        if self.state == OUTDATED:
            return f"{self.installed}  →  {self.latest} available"
        if self.state == UP_TO_DATE:
            return f"{self.installed} (current)"
        if self.state == NOT_NEEDED:
            return "not needed here"
        if self.state == UNCHECKED:
            # Some tools do not report a parseable version, and `installed`
            # then already carries the whole message.
            return (
                f"{self.installed} installed"
                if parse_version(self.installed)
                else self.installed
            )
        return f"{self.installed} — could not reach the index to compare"


def status_for(component: Component, settings=None, *, check_network: bool = False) -> Status:
    installed = installed_version(component, settings)
    latest = latest_version(component) if (check_network and installed) else ""
    detail = ""
    if component.key == "torch" and installed:
        cuda = torch_cuda_build()
        detail = f"CUDA {cuda} build" if cuda else "CPU-only build"
    state = compare(installed, latest)
    if state == MISSING and component.satisfied_by:
        stand_in = BY_KEY.get(component.satisfied_by)
        if stand_in and installed_version(stand_in, settings):
            return Status(component, "", "", NOT_NEEDED, f"{stand_in.title} is installed")
    if state == UNKNOWN and not check_network:
        # Not asked is not the same as asked and got no answer, and saying
        # "unknown" for something nobody has looked up yet reads as broken.
        state = UNCHECKED
    return Status(component, installed, latest, state, detail)


def statuses(settings=None, *, check_network: bool = False) -> list[Status]:
    """Every component's state. The network lookups run together, not in turn."""
    # Anything installed since this process started is on PATH in the registry
    # and nowhere else; without this the scan reports it as missing.
    refresh_path()
    if not check_network:
        return [status_for(c, settings) for c in COMPONENTS]

    from concurrent.futures import ThreadPoolExecutor

    # Seven indexes answered one after another is seven timeouts deep in the
    # worst case; asked at once it is one.
    with ThreadPoolExecutor(max_workers=len(COMPONENTS)) as pool:
        return list(
            pool.map(
                lambda c: status_for(c, settings, check_network=True), COMPONENTS
            )
        )


def missing(statuses_: list[Status], only_required: bool = False) -> list[Status]:
    out = [s for s in statuses_ if s.state == MISSING]
    return [s for s in out if not s.component.optional] if only_required else out


def outdated(statuses_: list[Status]) -> list[Status]:
    return [s for s in statuses_ if s.state == OUTDATED]


def run_command(
    cmd: list[str],
    *,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Run an install/update, streaming its output line by line.

    Returns the exit code. The caller decides what a non-zero one means: pip
    failing to build a wheel for this Python is a normal outcome here, not a
    crash.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if on_output:
            on_output(line.rstrip())
        if should_cancel and should_cancel():
            process.terminate()
            break
    process.wait()
    return process.returncode
