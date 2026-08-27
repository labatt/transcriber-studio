# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Version detection, update comparison, and the commands that close the gap."""

from __future__ import annotations

import sys

from transcriber_studio import components as C
from transcriber_studio.components import MISSING, OUTDATED, UNKNOWN, UP_TO_DATE

#: Windows paths for the command-building tests. Assembled from parts so that
#: no escape sequence can eat the separator — one of these spent a while
#: containing a literal vertical tab and silently comparing equal to its typo.
SEP = chr(92)
PY_EXE = "C:" + SEP + "py.exe"
LOG_PATH = "C:" + SEP + "log.txt"


# ---- comparing versions ------------------------------------------------


def test_nothing_installed_is_missing_whatever_the_index_says():
    assert C.compare("", "1.2.3") == MISSING
    assert C.compare("", "") == MISSING


def test_an_unreachable_index_is_unknown_not_up_to_date():
    """Reporting "current" when we could not ask would be the wrong answer."""
    assert C.compare("1.2.3", "") == UNKNOWN


def test_older_is_outdated_and_equal_is_current():
    assert C.compare("1.2.3", "1.2.4") == OUTDATED
    assert C.compare("1.2.3", "2.0.0") == OUTDATED
    assert C.compare("1.2.3", "1.2.3") == UP_TO_DATE


def test_a_newer_local_build_is_not_reported_as_behind():
    assert C.compare("4.9.0", "4.8.1") == UP_TO_DATE


def test_version_parts_compare_numerically_not_as_text():
    assert C.compare("1.9.0", "1.10.0") == OUTDATED
    assert C.parse_version("1.10.0") > C.parse_version("1.9.0")


def test_differently_long_versions_compare_by_value():
    assert C.compare("1.2", "1.2.0") == UP_TO_DATE
    assert C.compare("1.2", "1.2.1") == OUTDATED


def test_a_local_build_tag_does_not_make_a_version_look_different():
    """torch reports 2.6.0+cu124; the build tag says which wheel, not which release."""
    assert C.compare("2.6.0+cu124", "2.6.0") == UP_TO_DATE
    assert C.compare("2.6.0+cu124", "2.7.0") == OUTDATED


def test_an_unparseable_version_is_unknown_rather_than_a_wrong_answer():
    assert C.compare("installed", "0.5.6") == UNKNOWN


# ---- the commands ------------------------------------------------------


def test_pip_components_install_into_the_python_that_is_running():
    cmd = C.install_command(C.BY_KEY["faster-whisper"])

    assert cmd[:3] == [sys.executable, "-m", "pip"]
    assert cmd[-1] == "faster-whisper"
    assert "--upgrade" not in cmd


def test_updating_a_pip_component_upgrades_it():
    assert "--upgrade" in C.update_command(C.BY_KEY["faster-whisper"])


def test_a_cuda_torch_is_updated_as_a_matched_set_from_the_cuda_index(monkeypatch):
    """A plain pip upgrade would swap the CUDA build for the CPU one."""
    monkeypatch.setattr(C, "torch_cuda_build", lambda: "12.4")

    cmd = C.update_command(C.BY_KEY["torch"])

    assert cmd[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu124"]
    assert all(pkg in cmd for pkg in ("torchvision", "torchaudio", "torchcodec"))


def test_a_cpu_torch_stays_on_pypi_but_still_moves_as_a_set(monkeypatch):
    monkeypatch.setattr(C, "torch_cuda_build", lambda: "")

    cmd = C.update_command(C.BY_KEY["torch"])

    assert "--index-url" not in cmd
    assert cmd[-3:] == ["torchvision", "torchaudio", "torchcodec"]


def test_npm_and_winget_components_get_their_own_commands(monkeypatch):
    monkeypatch.setattr(C, "_platform_key", lambda: "win32")

    assert C.update_command(C.BY_KEY["plaud"])[1:] == [
        "install", "-g", "@plaud-ai/cli@latest"
    ]
    assert C.update_command(C.BY_KEY["ffmpeg"])[1:4] == [
        "upgrade", "--id", "Gyan.FFmpeg"
    ]


def test_winget_never_waits_for_a_prompt_nobody_can_answer(monkeypatch):
    """An interactive winget behind a piped stdout hangs until it is killed."""
    monkeypatch.setattr(C, "_platform_key", lambda: "win32")

    cmd = C.update_command(C.BY_KEY["ffmpeg"])

    assert "--disable-interactivity" in cmd
    assert "--accept-source-agreements" in cmd
    assert "--accept-package-agreements" in cmd


def test_a_downloaded_binary_has_no_command_to_run():
    """deep-filter is replaced by hand, so the UI must offer the page instead."""
    assert C.update_command(C.BY_KEY["deep-filter"]) == []
    assert C.install_command(C.BY_KEY["deep-filter"]) == []
    assert C.BY_KEY["deep-filter"].url


def test_command_text_quotes_only_what_needs_it():
    text = C.command_text([r"C:\Program Files\py.exe", "-m", "pip"])

    assert text == r'"C:\Program Files\py.exe" -m pip'


# ---- statuses ----------------------------------------------------------


def test_status_summary_says_which_versions_are_involved():
    component = C.BY_KEY["faster-whisper"]

    assert "not installed" in C.Status(component, "", "", MISSING).summary()
    assert "→" in C.Status(component, "1.0", "2.0", OUTDATED).summary()
    assert "current" in C.Status(component, "2.0", "2.0", UP_TO_DATE).summary()
    assert "could not reach" in C.Status(component, "2.0", "", UNKNOWN).summary()
    # Not asked yet is a different thing from asked and unanswered.
    assert C.Status(component, "2.0", "", C.UNCHECKED).summary() == "2.0 installed"


def test_only_missing_and_outdated_offer_an_action():
    component = C.BY_KEY["faster-whisper"]

    assert C.Status(component, "1.0", "2.0", OUTDATED).is_actionable
    assert C.Status(component, "", "", MISSING).is_actionable
    assert not C.Status(component, "2.0", "2.0", UP_TO_DATE).is_actionable
    assert not C.Status(component, "2.0", "", UNKNOWN).is_actionable
    assert not C.Status(component, "2.0", "", C.UNCHECKED).is_actionable


def test_scanning_without_the_network_asks_no_index(monkeypatch):
    """Opening the window must not wait on PyPI before it can draw."""
    monkeypatch.setattr(
        C, "latest_version",
        lambda _c: (_ for _ in ()).throw(AssertionError("network was used")),
    )

    statuses = C.statuses(check_network=False)

    assert statuses and all(s.latest == "" for s in statuses)
    # …and it says so, rather than claiming the version is unknowable.
    assert all(s.state != UNKNOWN for s in statuses)


def test_refreshing_path_only_ever_adds(monkeypatch):
    """An activated virtualenv lives on PATH and nowhere else.

    Holds on every platform: off Windows there is no registry to read and the
    call is a no-op, which satisfies "only ever adds" trivially.
    """
    import os

    # No drive letters here: a colon is the path separator on POSIX, so a
    # Windows-shaped fixture splits into pieces the moment this runs on Linux.
    before = [os.path.join("fake-venv", "Scripts"), "somewhere-else"]
    monkeypatch.setenv("PATH", os.pathsep.join(before))

    added = C.refresh_path()

    entries = os.environ["PATH"].split(os.pathsep)
    assert entries[: len(before)] == before      # nothing removed or reordered
    assert all(entry in entries for entry in added)


def test_a_permission_refusal_is_recognised():
    assert C.looks_like_a_permission_problem(
        "ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied"
    )
    assert C.looks_like_a_permission_problem("Consider using the `--user` option")
    assert not C.looks_like_a_permission_problem("Successfully installed foo-1.2.3")


def test_an_elevated_command_wraps_the_original_and_captures_its_output():
    cmd = C.elevated_command([PY_EXE, "-m", "pip", "install", "x"], LOG_PATH)
    text = C.command_text(cmd)

    assert cmd[0] == "powershell"
    assert "-Verb RunAs" in text and "-Wait" in text
    assert LOG_PATH in text          # output has to come back somehow
    assert "'-m', 'pip', 'install', 'x'" in text


def test_a_user_scope_install_needs_no_rights(monkeypatch):
    monkeypatch.setattr(C, "site_packages_writable", lambda: False)
    monkeypatch.setattr(C, "in_virtualenv", lambda: False)
    monkeypatch.setattr(C, "is_elevated", lambda: False)

    assert C.pip_scope() == "user"
    assert "--user" in C.install_command(C.BY_KEY["faster-whisper"])
    assert not C.needs_elevation(C.BY_KEY["faster-whisper"])


def test_an_unwritable_virtualenv_is_the_case_that_needs_rights(monkeypatch):
    """--user is invalid inside a venv, so there is nothing left but elevation."""
    monkeypatch.setattr(C, "site_packages_writable", lambda: False)
    monkeypatch.setattr(C, "in_virtualenv", lambda: True)
    monkeypatch.setattr(C, "is_elevated", lambda: False)

    assert C.pip_scope() == "elevated"
    assert "--user" not in C.install_command(C.BY_KEY["faster-whisper"])
    assert C.needs_elevation(C.BY_KEY["faster-whisper"])


def test_a_component_that_cannot_be_reached_still_reports_what_is_installed(monkeypatch):
    monkeypatch.setattr(C, "installed_version", lambda _c, _s=None: "1.0.0")
    monkeypatch.setattr(C, "latest_version", lambda _c: "")

    status = C.status_for(C.BY_KEY["faster-whisper"], check_network=True)

    assert status.installed == "1.0.0"
    assert status.state == UNKNOWN


def test_torch_reports_which_build_is_installed(monkeypatch):
    monkeypatch.setattr(C, "installed_version", lambda _c, _s=None: "2.6.0+cu124")
    monkeypatch.setattr(C, "torch_cuda_build", lambda: "12.4")

    assert "CUDA 12.4" in C.status_for(C.BY_KEY["torch"]).detail

    monkeypatch.setattr(C, "torch_cuda_build", lambda: "")
    assert "CPU-only" in C.status_for(C.BY_KEY["torch"]).detail


def test_faster_whisper_is_detected_in_this_environment():
    """It is a hard dependency of the local engine, so it is really installed."""
    status = C.status_for(C.BY_KEY["faster-whisper"])

    assert status.installed
    assert status.state != MISSING


def test_missing_and_outdated_split_the_list():
    component = C.BY_KEY["faster-whisper"]
    statuses = [
        C.Status(component, "", "", MISSING),
        C.Status(component, "1.0", "2.0", OUTDATED),
        C.Status(component, "2.0", "2.0", UP_TO_DATE),
    ]

    assert len(C.missing(statuses)) == 1
    assert len(C.outdated(statuses)) == 1


def test_required_components_can_be_singled_out():
    ffmpeg = C.BY_KEY["ffmpeg"]
    optional = C.BY_KEY["deepfilternet"]
    statuses = [C.Status(ffmpeg, "", "", MISSING), C.Status(optional, "", "", MISSING)]

    assert [s.component.key for s in C.missing(statuses, only_required=True)] == ["ffmpeg"]


def test_windows_cli_shims_are_resolved_before_being_run(monkeypatch):
    """subprocess cannot find plaud.CMD by bare name the way a shell can."""
    monkeypatch.setattr(C.shutil, "which", lambda name: rf"C:\npm\{name}.CMD")

    assert C.executable("npm") == r"C:\npm\npm.CMD"
    assert C.update_command(C.BY_KEY["plaud"])[0] == r"C:\npm\npm.CMD"


def test_an_unresolvable_cli_falls_back_to_the_bare_name(monkeypatch):
    monkeypatch.setattr(C.shutil, "which", lambda _name: None)

    assert C.executable("npm") == "npm"


def test_a_component_with_a_stand_in_is_not_reported_as_missing(monkeypatch):
    """The pip package and the binary are the same denoiser; one is enough."""
    monkeypatch.setattr(
        C, "installed_version",
        lambda c, _s=None: "0.5.6" if c.key == "deep-filter" else "",
    )

    status = C.status_for(C.BY_KEY["deepfilternet"])

    assert status.state == C.NOT_NEEDED
    assert not status.is_actionable
    assert "deep-filter" in status.detail.lower()
    assert C.missing([status]) == []


def test_it_is_still_missing_when_the_stand_in_is_absent_too(monkeypatch):
    monkeypatch.setattr(C, "installed_version", lambda _c, _s=None: "")

    assert C.status_for(C.BY_KEY["deepfilternet"]).state == MISSING


# ---- other platforms ---------------------------------------------------
# The app is developed on Windows; these check that the decisions it makes for
# macOS and Linux are the right ones, since the GUI cannot be exercised here.


def test_ffmpeg_is_installed_with_each_platforms_own_package_manager(monkeypatch):
    ffmpeg = C.BY_KEY["ffmpeg"]

    for platform, manager, verb in (
        ("win32", "winget", "install"),
        ("darwin", "brew", "install"),
        ("linux", "apt-get", "install"),
    ):
        monkeypatch.setattr(C, "_platform_key", lambda p=platform: p)
        monkeypatch.setattr(C, "executable", lambda name: name)
        cmd = C.install_command(ffmpeg)
        assert cmd[0] == manager, platform
        assert verb in cmd, platform


def test_a_platform_with_no_package_manager_entry_gets_no_command(monkeypatch):
    """Better to show the download page than a command that does not exist here."""
    monkeypatch.setattr(C, "_platform_key", lambda: "freebsd14")

    assert C.install_command(C.BY_KEY["ffmpeg"]) == []
    assert C.BY_KEY["ffmpeg"].url        # …which is what the UI falls back to


def test_only_windows_offers_to_elevate_a_command_itself(monkeypatch):
    """An untested privilege-escalation path is not something to ship."""
    monkeypatch.setattr(C, "_platform_key", lambda: "win32")
    assert C.can_elevate()

    for platform in ("darwin", "linux"):
        monkeypatch.setattr(C, "_platform_key", lambda p=platform: p)
        assert not C.can_elevate(), platform


def test_apt_is_known_to_need_root_where_brew_and_winget_do_not(monkeypatch):
    monkeypatch.setattr(C, "is_elevated", lambda: False)
    ffmpeg = C.BY_KEY["ffmpeg"]

    monkeypatch.setattr(C, "_platform_key", lambda: "linux")
    assert C.needs_elevation(ffmpeg)

    for platform in ("darwin", "win32"):
        monkeypatch.setattr(C, "_platform_key", lambda p=platform: p)
        assert not C.needs_elevation(ffmpeg), platform


def test_refresh_path_is_a_windows_concern_and_no_op_elsewhere(monkeypatch):
    monkeypatch.setattr(C, "_platform_key", lambda: "linux")
    assert C.refresh_path() == []


def test_a_tool_without_a_parseable_version_is_not_reported_twice(monkeypatch):
    """deep-filter reports "installed" when -V gives nothing; "installed
    installed" was what that used to render as."""
    status = C.Status(C.BY_KEY["deep-filter"], "installed", "", C.UNCHECKED)

    assert status.summary() == "installed"
    assert C.Status(C.BY_KEY["ffmpeg"], "9.0.1", "", C.UNCHECKED).summary() == "9.0.1 installed"
