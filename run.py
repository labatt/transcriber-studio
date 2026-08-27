#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launcher for Transcriber Studio, for running from a source checkout.

    python run.py

An installed copy gets a `transcriber-studio` command instead.
"""
from transcriber_studio.suppress_warnings import configure as _configure_warnings

_configure_warnings()

from transcriber_studio.__main__ import main

if __name__ == "__main__":
    main()
