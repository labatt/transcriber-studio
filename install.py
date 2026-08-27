#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Install everything Transcriber Studio needs, on whichever OS you are on.

    python install.py              # check, explain, then install with confirmation
    python install.py --check      # report only, change nothing
    python install.py --yes        # no prompts, for a scripted setup
    python install.py --minimal    # required pieces only, skip the optional ones

Written against the standard library alone, because it runs before anything is
installed. It reuses transcriber_studio.components for the registry, version
detection and command building — that module is import-safe on a bare Python
for exactly this reason, so the installer and the app's own Components window
can never disagree about what "installed" means.

What it cannot do is install the Python it is running on. `install.ps1` and
`install.sh` handle that and then hand over to this script.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transcriber_studio import components as C  # noqa: E402

MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = "3.13"

#: Where a manual ffmpeg lands on Windows when there is no package manager.
FFMPEG_WINDOWS_ZIP = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
PYTHON_WINDOWS_INSTALLER = (
    "https://www.python.org/ftp/python/{v}/python-{v}-amd64.exe"
)

OK, WARN, BAD, DOT = "[ok]", "[!]", "[x]", " - "
if os.name != "nt" or os.environ.get("WT_SESSION"):
    OK, WARN, BAD, DOT = "✓", "!", "✗", "·"


# ---------------------------------------------------------------- output
class Report:
    """Everything that happened, so the end of the run can summarise it."""

    def __init__(self) -> None:
        self.done: list[str] = []
        self.skipped: list[str] = []
        self.problems: list[tuple[str, str]] = []

    def ok(self, message: str) -> None:
        print(f"  {OK} {message}")
        self.done.append(message)

    def skip(self, message: str) -> None:
        print(f"  {DOT} {message}")
        self.skipped.append(message)

    def warn(self, message: str, fix: str = "") -> None:
        print(f"  {WARN} {message}")
        if fix:
            print(f"      → {fix}")
        self.problems.append((message, fix))

    def fail(self, message: str, fix: str = "") -> None:
        print(f"  {BAD} {message}")
        if fix:
            print(f"      → {fix}")
        self.problems.append((message, fix))


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# ------------------------------------------------------------ the machine
@dataclass(frozen=True)
class Machine:
    """What we are installing onto."""

    system: str          # windows | macos | linux
    manager: str         # winget | brew | apt | dnf | pacman | ""
    python: tuple[int, int]
    in_venv: bool
    gpu: str = ""        # GPU name, "" when there is none
    driver_cuda: str = ""  # highest CUDA the driver supports

    @property
    def python_ok(self) -> bool:
        return self.python >= MIN_PYTHON

    def describe(self) -> list[str]:
        rows = [
            f"OS              {platform.platform()}",
            f"Package manager {self.manager or 'none found'}",
            f"Python          {'.'.join(map(str, self.python))}"
            + ("" if self.python_ok else f"  (too old, {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ needed)"),
            f"Virtualenv      {'yes' if self.in_venv else 'no — installing into the system Python'}",
        ]
        rows.append(
            f"GPU             {self.gpu} (driver supports CUDA {self.driver_cuda})"
            if self.gpu else "GPU             none detected — CPU or a cloud engine"
        )
        return rows


def detect_manager(system: str) -> str:
    """The package manager to drive, or "" when there is none to drive."""
    candidates = {
        "windows": ["winget"],
        "macos": ["brew"],
        "linux": ["apt-get", "dnf", "pacman", "zypper"],
    }[system]
    for name in candidates:
        if shutil.which(name):
            return "apt" if name == "apt-get" else name
    return ""


def detect_gpu() -> tuple[str, str]:
    """(GPU name, highest CUDA the driver supports). ("", "") when there is none."""
    if not shutil.which("nvidia-smi"):
        return "", ""
    try:
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        banner = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return "", ""
    import re

    cuda = re.search(r"CUDA Version:\s*([\d.]+)", banner)
    return (name[0] if name else "NVIDIA GPU"), (cuda.group(1) if cuda else "")


def look_around() -> Machine:
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    gpu, driver = detect_gpu()
    return Machine(
        system=system,
        manager=detect_manager(system),
        python=sys.version_info[:2],
        in_venv=sys.prefix != sys.base_prefix,
        gpu=gpu,
        driver_cuda=driver,
    )


# ------------------------------------------------------------------ steps
@dataclass
class Step:
    key: str
    title: str
    why: str
    required: bool = False
    #: Reasons this step does not apply to this machine.
    skip_if: str = ""
    extras: list[str] = field(default_factory=list)


def plan(machine: Machine, *, minimal: bool, want_gpu: bool) -> list[Step]:
    """The steps for this machine, in the order they have to happen."""
    steps = [
        Step("ffmpeg", "ffmpeg",
             "Decodes audio, splits channels, and is the fallback denoiser.",
             required=True),
        Step("app", "Transcriber Studio and the local engine",
             "The app itself, PySide6, and faster-whisper."),
    ]
    if want_gpu and machine.gpu:
        steps.append(Step("torch", "PyTorch (CUDA build)",
                          "Runs speaker detection on the GPU instead of the CPU."))
    if not minimal:
        steps += [
            Step("diarization", "Speaker diarization",
                 "pyannote, so transcripts say who spoke."),
            Step("deep-filter", "DeepFilterNet denoiser",
                 "The biggest accuracy win on difficult audio."),
            Step("plaud", "PLAUD CLI",
                 "Imports recordings from a PLAUD cloud account.",
                 skip_if="" if shutil.which("npm") else
                         "needs Node.js 20+; install Node first if you want PLAUD import"),
        ]
    return steps


# -------------------------------------------------------------- executing
class Installer:
    def __init__(self, machine: Machine, report: Report, *, assume_yes: bool, dry_run: bool):
        self.m = machine
        self.r = report
        self.assume_yes = assume_yes
        self.dry_run = dry_run

    # ---- asking -------------------------------------------------------
    def confirm(self, question: str) -> bool:
        if self.dry_run:
            # Answer yes so the commands are printed: the point of a dry run is
            # seeing what would happen, not seeing everything declined.
            print(f"  {question} [would ask]")
            return True
        if self.assume_yes:
            return True
        try:
            return input(f"  {question} [Y/n] ").strip().lower() in ("", "y", "yes")
        except EOFError:
            return False

    def run(self, cmd: list[str], what: str) -> bool:
        """Run a command, showing it first. Returns whether it succeeded."""
        print(f"      $ {C.command_text(cmd)}")
        if self.dry_run:
            return True
        try:
            code = C.run_command(cmd, on_output=lambda line: print(f"        {line}"))
        except FileNotFoundError:
            self.r.fail(f"{what}: {cmd[0]} is not on PATH.")
            return False
        except Exception as exc:                       # noqa: BLE001 - report anything
            self.r.fail(f"{what} could not run: {exc}")
            return False
        if code != 0:
            self.r.fail(
                f"{what} failed (exit {code}).",
                "The output above says why. Nothing else was changed."
                if not C.looks_like_a_permission_problem("")
                else "Looks like a permissions problem — see the note below.",
            )
            return False
        return True

    # ---- downloads ----------------------------------------------------
    def download(self, url: str, dest: Path, what: str) -> bool:
        print(f"      ↓ {url}")
        if self.dry_run:
            return True
        try:
            request = urllib.request.Request(url, headers={"User-Agent": C.USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                total = int(response.headers.get("Content-Length") or 0)
                dest.parent.mkdir(parents=True, exist_ok=True)
                got = 0
                with open(dest, "wb") as handle:
                    while chunk := response.read(256 * 1024):
                        handle.write(chunk)
                        got += len(chunk)
                        if total:
                            print(f"\r        {got * 100 // total}%", end="", flush=True)
                if total:
                    print("\r        100%")
            return True
        except Exception as exc:                       # noqa: BLE001
            self.r.fail(f"{what}: download failed ({exc})", f"Fetch it by hand from {url}")
            return False

    # ---- the steps ----------------------------------------------------
    def do_ffmpeg(self) -> None:
        component = C.BY_KEY["ffmpeg"]
        installed = C.installed_version(component)
        if installed:
            latest = C.latest_version(component)
            if latest and C.compare(installed, latest) == C.OUTDATED:
                print(f"      ffmpeg {installed} installed, {latest} available")
                if self.confirm("Upgrade it?"):
                    cmd = C.update_command(component, latest)
                    if cmd and self.run(cmd, "ffmpeg upgrade"):
                        C.refresh_path()
                        self.r.ok(f"ffmpeg upgraded to {C.installed_version(component) or latest}")
                        return
                self.r.skip(f"ffmpeg {installed} kept")
            else:
                self.r.ok(f"ffmpeg {installed} already current")
            return

        cmd = C.install_command(component)
        if cmd and self.confirm(f"Install ffmpeg with {self.m.manager}?"):
            if self.run(cmd, "ffmpeg install"):
                C.refresh_path()
                if C.installed_version(component):
                    self.r.ok(f"ffmpeg {C.installed_version(component)} installed")
                else:
                    self.r.warn(
                        "ffmpeg installed but not on PATH yet.",
                        "Open a new terminal, or restart, and run --check again.",
                    )
                return
        if self.m.system == "windows":
            self.install_ffmpeg_windows_zip()
            return
        self.r.fail(
            "ffmpeg is required and could not be installed automatically.",
            {
                "macos": "Install Homebrew from https://brew.sh, then: brew install ffmpeg",
                "linux": "Install it with your distribution's package manager, e.g. "
                         "sudo apt install ffmpeg",
            }.get(self.m.system, "See https://ffmpeg.org/download.html"),
        )

    def install_ffmpeg_windows_zip(self) -> None:
        """No winget: fetch a static build and put it beside the app's own data."""
        print("      No package manager found — fetching a static build instead.")
        if not self.confirm("Download ffmpeg (~80 MB) and install it for this user?"):
            self.r.fail("ffmpeg is required.", f"Download it from {FFMPEG_WINDOWS_ZIP}")
            return
        from transcriber_studio.config import APP_DIR

        target = APP_DIR / "bin"
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ffmpeg.zip"
            if not self.download(FFMPEG_WINDOWS_ZIP, archive, "ffmpeg"):
                return
            if self.dry_run:
                return
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if member.endswith((".exe",)) and "/bin/" in member:
                        source = zf.open(member)
                        with open(target / Path(member).name, "wb") as out:
                            shutil.copyfileobj(source, out)
        if (target / "ffmpeg.exe").exists():
            self.add_to_user_path(target)
            self.r.ok(f"ffmpeg installed into {target}")
        else:
            self.r.fail("ffmpeg archive did not contain the expected binaries.")

    def add_to_user_path(self, directory: Path) -> None:
        """Put a directory on the user's PATH, for this process and for future ones."""
        os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
        if self.m.system != "windows" or self.dry_run:
            print(f"      Add {directory} to your PATH to make this permanent.")
            return
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as key:
                current, kind = winreg.QueryValueEx(key, "Path")
                if str(directory).lower() in current.lower():
                    return
                winreg.SetValueEx(key, "Path", 0, kind,
                                  f"{current}{os.pathsep}{directory}")
            print(f"      Added {directory} to your PATH (new terminals will see it).")
        except OSError as exc:
            print(f"      Could not update PATH ({exc}); add {directory} by hand.")

    def do_app(self) -> None:
        extras = "local"
        cmd = [*C.PIP, "install", "-e", f".[{extras}]"]
        if C.pip_scope() == "user" and not C.is_elevated():
            cmd.append("--user")
        if not self.confirm("Install the app and the local Whisper engine?"):
            self.r.skip("app dependencies not installed")
            return
        if self.run(cmd, "app install"):
            self.r.ok("app and local engine installed")

    def do_torch(self) -> None:
        component = C.BY_KEY["torch"]
        installed = C.installed_version(component)
        cuda_build = C.torch_cuda_build()
        if installed and cuda_build:
            self.r.ok(f"PyTorch {installed} already a CUDA build")
            return
        latest = C.latest_version(component) or ""
        channel = C.torch_channel_for(latest) if latest else ""
        if not channel:
            # torch_channel_for needs an installed CUDA torch to pick a channel;
            # with none, fall back to the newest CUDA 12 line the driver allows.
            channel = self.channel_for_driver()
        cmd = [*C.PIP, "install", "--upgrade", "torch", "torchvision", "torchaudio",
               "torchcodec", "--index-url", f"https://download.pytorch.org/whl/{channel}"]
        print(f"      Driver supports CUDA {self.m.driver_cuda}; using the {channel} wheels.")
        if installed and not cuda_build:
            print("      A CPU-only PyTorch is installed; this replaces it with the GPU build.")
        if not self.confirm("Install the CUDA build of PyTorch (a few GB)?"):
            self.r.skip("PyTorch left as it is — speaker detection will use the CPU")
            return
        if self.run(cmd, "PyTorch install"):
            self.r.ok(f"PyTorch installed from {channel}")

    def channel_for_driver(self) -> str:
        """The newest CUDA 12 channel this driver can run.

        CUDA 12 rather than 13 on purpose: CTranslate2, which runs Whisper, is
        built against CUDA 12 and cuDNN 9.
        """
        try:
            major = int((self.m.driver_cuda or "12").split(".")[0])
        except ValueError:
            major = 12
        return "cu126" if major >= 12 else "cu121"

    def do_diarization(self) -> None:
        if not self.confirm("Install speaker diarization (pyannote)?"):
            self.r.skip("diarization not installed")
            return
        cmd = [*C.PIP, "install", "-e", ".[diarization]"]
        if C.pip_scope() == "user" and not C.is_elevated():
            cmd.append("--user")
        if self.run(cmd, "diarization install"):
            self.r.ok("diarization installed")
            self.r.warn(
                "Diarization also needs a HuggingFace token and three accepted licences.",
                "The app's setup wizard walks through it; its Test button checks all three.",
            )

    def do_deep_filter(self) -> None:
        component = C.BY_KEY["deep-filter"]
        if C.installed_version(component):
            self.r.ok("DeepFilterNet binary already installed")
            return
        asset = self.deep_filter_asset()
        if not asset:
            self.r.warn(
                "No DeepFilterNet build for this platform.",
                "The app falls back to ffmpeg's denoiser, which is weaker but real.",
            )
            return
        if not self.confirm("Download the DeepFilterNet denoiser (~27 MB)?"):
            self.r.skip("denoising will fall back to ffmpeg")
            return
        from transcriber_studio.config import APP_DIR, load, save

        target = APP_DIR / "bin" / ("deep-filter.exe" if self.m.system == "windows"
                                    else "deep-filter")
        if not self.download(asset, target, "DeepFilterNet"):
            return
        if self.dry_run:
            return
        if self.m.system != "windows":
            target.chmod(0o755)
        settings = load()
        settings.deep_filter_path = str(target)
        settings.denoise_enabled = True
        save(settings)
        self.r.ok(f"DeepFilterNet installed and enabled ({target})")

    def deep_filter_asset(self) -> str:
        """The release asset matching this machine, or "" if there is none."""
        suffix = deep_filter_suffix(self.m.system, normalised_arch())
        if not suffix:
            return ""
        try:
            import json

            data = json.loads(C.fetch(
                "https://api.github.com/repos/Rikorose/DeepFilterNet/releases/latest"
            ))
        except Exception:
            return ""
        for item in data.get("assets", []):
            if item.get("name", "").endswith(suffix):
                return item.get("browser_download_url", "")
        return ""

    def do_plaud(self) -> None:
        component = C.BY_KEY["plaud"]
        installed = C.installed_version(component)
        if installed:
            self.r.ok(f"PLAUD CLI {installed} installed")
            return
        if not shutil.which("npm"):
            self.r.skip("PLAUD CLI skipped — Node.js 20+ is not installed")
            return
        if not self.confirm("Install the PLAUD CLI (for cloud import)?"):
            self.r.skip("PLAUD import not set up")
            return
        if self.run(C.install_command(component), "PLAUD CLI install"):
            self.r.ok("PLAUD CLI installed — sign in from the app's setup wizard")


#: Which DeepFilterNet release asset belongs to which machine. Downloading the
#: Windows .exe onto a Linux box is the failure this table exists to prevent.
DEEP_FILTER_ASSETS = {
    ("windows", "amd64"): "x86_64-pc-windows-msvc.exe",
    ("macos", "arm64"): "aarch64-apple-darwin",
    ("macos", "x86_64"): "x86_64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
}


def deep_filter_suffix(system: str, arch: str) -> str:
    """The release asset suffix for a platform, or "" when there is no build."""
    return DEEP_FILTER_ASSETS.get((system, arch), "")


def normalised_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "amd64" if sys.platform == "win32" else "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64" if sys.platform == "darwin" else "aarch64"
    return machine


# ------------------------------------------------------------------- main
def check_only(machine: Machine, report: Report) -> None:
    heading("What is installed")
    for status in C.statuses(check_network=True):
        line = f"{status.component.title:38} {status.summary()}"
        if status.state == C.MISSING and not status.component.optional:
            report.fail(line)
        elif status.state == C.MISSING:
            report.skip(line + "  (optional)")
        elif status.state == C.OUTDATED:
            report.warn(line)
        else:
            report.ok(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install everything Transcriber Studio needs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check", action="store_true",
                        help="report what is installed and stop")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="answer yes to everything")
    parser.add_argument("--minimal", action="store_true",
                        help="only what the app cannot run without")
    parser.add_argument("--no-gpu", action="store_true",
                        help="skip the CUDA PyTorch install even if a GPU is present")
    parser.add_argument("--dry-run", action="store_true",
                        help="show every command without running any of them")
    args = parser.parse_args(argv)

    print("Transcriber Studio — installer")
    report = Report()
    C.refresh_path()
    machine = look_around()

    heading("This machine")
    for row in machine.describe():
        print(f"  {row}")

    if not machine.python_ok:
        print()
        report.fail(
            f"Python {'.'.join(map(str, machine.python))} is too old "
            f"(need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+).",
            f"Install Python {RECOMMENDED_PYTHON} and run this again with it. "
            "install.ps1 (Windows) and install.sh (macOS/Linux) do that for you.",
        )
        return summarise(report)

    if args.check:
        check_only(machine, report)
        return summarise(report)

    if not machine.in_venv:
        print("\n  Note: no virtualenv is active, so this installs into the system"
              "\n  Python. That is fine, but a venv keeps it separate from your other"
              "\n  projects:  python -m venv .venv  &&  "
              + (".venv\\Scripts\\Activate.ps1" if machine.system == "windows"
                 else "source .venv/bin/activate"))

    steps = plan(machine, minimal=args.minimal, want_gpu=not args.no_gpu)
    heading("Plan")
    for step in steps:
        mark = "required" if step.required else "optional"
        print(f"  {DOT} {step.title}  ({mark})")
        print(f"      {step.why}")
        if step.skip_if:
            print(f"      note: {step.skip_if}")
    if args.dry_run:
        print("\n  --dry-run: showing commands only, nothing will be installed.")

    installer = Installer(machine, report, assume_yes=args.yes, dry_run=args.dry_run)
    actions = {
        "ffmpeg": installer.do_ffmpeg,
        "app": installer.do_app,
        "torch": installer.do_torch,
        "diarization": installer.do_diarization,
        "deep-filter": installer.do_deep_filter,
        "plaud": installer.do_plaud,
    }
    for step in steps:
        heading(step.title)
        try:
            actions[step.key]()
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            return summarise(report)
        except Exception as exc:                       # noqa: BLE001
            report.fail(f"{step.title}: unexpected error ({exc})")

    return summarise(report)


def summarise(report: Report) -> int:
    heading("Summary")
    print(f"  {len(report.done)} done, {len(report.skipped)} skipped, "
          f"{len(report.problems)} needing attention")
    for message, fix in report.problems:
        print(f"\n  {WARN} {message}")
        if fix:
            print(f"      → {fix}")
    if not report.problems:
        print("\n  Nothing outstanding. Start the app with:  python run.py")
        return 0
    print("\n  Re-run `python install.py --check` after fixing anything above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
