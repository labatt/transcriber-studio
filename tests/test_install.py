# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""The installer's decisions: which steps, which commands, which download."""

from __future__ import annotations

import install
from install import Machine, Report


def _machine(**over) -> Machine:
    base = dict(system="windows", manager="winget", python=(3, 13), in_venv=True)
    base.update(over)
    return Machine(**base)


# ---- reading the machine ----------------------------------------------


def test_each_os_looks_for_its_own_package_managers(monkeypatch):
    seen: list[str] = []

    def fake_which(name):
        seen.append(name)
        return f"/usr/bin/{name}" if name in ("brew", "dnf") else None

    monkeypatch.setattr(install.shutil, "which", fake_which)

    assert install.detect_manager("macos") == "brew"
    assert install.detect_manager("linux") == "dnf"        # apt-get absent, dnf present
    assert install.detect_manager("windows") == ""         # winget absent
    assert "winget" in seen and "apt-get" in seen


def test_apt_get_is_reported_as_apt(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda n: "/usr/bin/apt-get" if n == "apt-get" else None)

    assert install.detect_manager("linux") == "apt"


def test_a_python_below_the_floor_is_refused():
    assert not _machine(python=(3, 9)).python_ok
    assert _machine(python=(3, 10)).python_ok
    assert _machine(python=(3, 14)).python_ok


def test_no_nvidia_smi_means_no_gpu(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda _n: None)

    assert install.detect_gpu() == ("", "")


# ---- planning ----------------------------------------------------------


def test_a_machine_with_no_gpu_is_not_offered_cuda_pytorch():
    steps = install.plan(_machine(gpu=""), minimal=False, want_gpu=True)

    assert "torch" not in [s.key for s in steps]


def test_a_gpu_machine_is_offered_cuda_pytorch():
    steps = install.plan(_machine(gpu="RTX 4090"), minimal=False, want_gpu=True)

    assert "torch" in [s.key for s in steps]


def test_no_gpu_flag_wins_over_a_present_gpu():
    steps = install.plan(_machine(gpu="RTX 4090"), minimal=False, want_gpu=False)

    assert "torch" not in [s.key for s in steps]


def test_minimal_keeps_only_what_the_app_cannot_run_without():
    steps = install.plan(_machine(gpu="RTX 4090"), minimal=True, want_gpu=True)
    keys = [s.key for s in steps]

    assert "ffmpeg" in keys and "app" in keys
    assert "diarization" not in keys and "deep-filter" not in keys and "plaud" not in keys


def test_ffmpeg_is_the_one_required_step():
    steps = install.plan(_machine(), minimal=False, want_gpu=False)

    assert [s.key for s in steps if s.required] == ["ffmpeg"]


def test_ffmpeg_comes_before_anything_that_needs_it():
    """The app's own tests decode audio, so ffmpeg cannot come second."""
    keys = [s.key for s in install.plan(_machine(), minimal=False, want_gpu=False)]

    assert keys.index("ffmpeg") == 0


# ---- the download that platform-detection has to get right -------------


def test_each_platform_gets_its_own_denoiser_build():
    assert install.deep_filter_suffix("windows", "amd64").endswith(".exe")
    assert "linux" in install.deep_filter_suffix("linux", "x86_64")
    assert "darwin" in install.deep_filter_suffix("macos", "arm64")
    assert "darwin" in install.deep_filter_suffix("macos", "x86_64")


def test_a_windows_binary_is_never_chosen_for_a_unix_machine():
    """The bug this table exists to prevent."""
    for system, arch in (("linux", "x86_64"), ("macos", "arm64"), ("linux", "aarch64")):
        assert not install.deep_filter_suffix(system, arch).endswith(".exe")


def test_a_platform_with_no_build_gets_nothing_rather_than_the_wrong_thing():
    assert install.deep_filter_suffix("freebsd", "x86_64") == ""
    assert install.deep_filter_suffix("linux", "riscv64") == ""


# ---- the CUDA channel --------------------------------------------------


def _installer(machine: Machine) -> install.Installer:
    return install.Installer(machine, Report(), assume_yes=True, dry_run=True)


def test_a_cuda_12_or_newer_driver_gets_the_cuda_12_wheels():
    """CUDA 12 on purpose: CTranslate2 is built against it."""
    for driver in ("12.4", "13.2", "12.0"):
        assert _installer(_machine(driver_cuda=driver)).channel_for_driver() == "cu126"


def test_an_older_driver_falls_back_rather_than_failing():
    assert _installer(_machine(driver_cuda="11.8")).channel_for_driver() == "cu121"


def test_an_unreadable_driver_version_does_not_crash_the_run():
    assert _installer(_machine(driver_cuda="")).channel_for_driver() == "cu126"
    assert _installer(_machine(driver_cuda="unknown")).channel_for_driver() == "cu126"


# ---- reporting ---------------------------------------------------------


def test_the_summary_counts_what_happened(capsys):
    report = Report()
    report.ok("did a thing")
    report.skip("skipped a thing")
    report.fail("broke", "try this")

    code = install.summarise(report)
    out = capsys.readouterr().out

    assert code == 1                       # a problem means a non-zero exit
    assert "1 done, 1 skipped, 1 needing attention" in out
    assert "try this" in out               # the fix is repeated at the end


def test_a_clean_run_exits_zero_and_says_what_to_do_next(capsys):
    code = install.summarise(Report())

    assert code == 0
    assert "python run.py" in capsys.readouterr().out


def test_a_dry_run_never_executes_anything(capsys):
    installer = _installer(_machine())

    assert installer.run(["definitely-not-a-real-command"], "test") is True
    assert "definitely-not-a-real-command" in capsys.readouterr().out
