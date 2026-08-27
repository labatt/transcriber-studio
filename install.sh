#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Set up Transcriber Studio on macOS or Linux, starting from nothing.
#
#   ./install.sh              # install everything, asking as it goes
#   ./install.sh --check      # report what is installed, change nothing
#   ./install.sh --yes        # no prompts
#
# Finds a Python new enough to run the installer, installs one if there is
# none, then hands over to install.py which does the rest. This exists because
# install.py cannot install the Python it is running on.

set -euo pipefail

MIN_MAJOR=3
MIN_MINOR=10
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say()  { printf '\n%s\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
note() { printf '      %s\n' "$1"; }

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        --dry-run) DRY_RUN=1 ;;
    esac
done

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    [ "$DRY_RUN" = 1 ] && return 0
    read -r -p "  $1 [Y/n] " answer
    [ -z "$answer" ] || [ "$answer" = "y" ] || [ "$answer" = "Y" ]
}

printf 'Transcriber Studio — setup\n'

# --- what are we on? ------------------------------------------------------
case "$(uname -s)" in
    Darwin) SYSTEM=macos ;;
    Linux)  SYSTEM=linux ;;
    *)      SYSTEM=unknown ;;
esac

MANAGER=""
for candidate in brew apt-get dnf pacman zypper; do
    if command -v "$candidate" >/dev/null 2>&1; then MANAGER="$candidate"; break; fi
done

say "This machine"
note "OS              $(uname -s) $(uname -r)"
note "Package manager ${MANAGER:-none found}"

# --- find a Python we can use --------------------------------------------
find_python() {
    for exe in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
        command -v "$exe" >/dev/null 2>&1 || continue
        if "$exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
            command -v "$exe"
            return 0
        fi
    done
    return 1
}

say "Python"
if PYTHON="$(find_python)"; then
    ok "$("$PYTHON" -V 2>&1) at $PYTHON"
else
    bad "No Python ${MIN_MAJOR}.${MIN_MINOR} or newer found."
    case "$MANAGER" in
        brew)    INSTALL_CMD="brew install python@3.13" ;;
        apt-get) INSTALL_CMD="sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" ;;
        dnf)     INSTALL_CMD="sudo dnf install -y python3 python3-pip" ;;
        pacman)  INSTALL_CMD="sudo pacman -S --noconfirm python python-pip" ;;
        zypper)  INSTALL_CMD="sudo zypper install -y python3 python3-pip" ;;
        *)       INSTALL_CMD="" ;;
    esac
    if [ -z "$INSTALL_CMD" ]; then
        note "No package manager was found, so Python cannot be installed for you."
        note "Install Python ${MIN_MAJOR}.${MIN_MINOR}+ from https://www.python.org/downloads/"
        note "and run this again."
        exit 1
    fi
    note "Would run: $INSTALL_CMD"
    if confirm "Install Python now?"; then
        if [ "$DRY_RUN" = 1 ]; then
            note "(dry run — not installed)"
            exit 0
        fi
        eval "$INSTALL_CMD"
        PYTHON="$(find_python)" || { bad "Python still not found after installing."; exit 1; }
        ok "$("$PYTHON" -V 2>&1) installed"
    else
        bad "Python is required."
        exit 1
    fi
fi

# Debian splits venv and pip out of the base package, and the failure only
# shows up much later as a confusing pip error.
if [ "$MANAGER" = "apt-get" ]; then
    if ! "$PYTHON" -c "import ensurepip, venv" >/dev/null 2>&1; then
        note "python3-venv is missing (Debian and Ubuntu package it separately)."
        if confirm "Install python3-venv and python3-pip?"; then
            [ "$DRY_RUN" = 1 ] || sudo apt-get install -y python3-venv python3-pip
        fi
    fi
fi

# --- hand over ------------------------------------------------------------
say "Handing over to install.py"
exec "$PYTHON" "$HERE/install.py" "$@"
