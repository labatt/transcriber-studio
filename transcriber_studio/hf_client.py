# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""HuggingFace token checks for the pyannote models speaker detection needs.

A token being valid is only half the story: the pyannote models are gated, so a
perfectly good token still fails at run time until the user has clicked "Agree
and access" on each model page. Both are checked here, because the difference
is invisible until a job dies halfway through.
"""

from __future__ import annotations

import requests

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
MODEL_URL = "https://huggingface.co/api/models/{repo}"
# Gating bites on file downloads, not on metadata: /api/models answers 200 for
# a gated repo even unauthenticated, so asking it proves nothing. Reaching for
# a file is what returns 401/403 GatedRepo until the licence is accepted.
FILE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
PROBE_FILE = "config.yaml"      # present in all three pyannote repos

#: The gated repos pyannote loads. Access is granted per repo, per account.
GATED_REPOS = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-3.1",
]

TIMEOUT = 30


class HFError(RuntimeError):
    """A token that will not work for diarization, and why."""


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token.strip()}"}


def _repo_accessible(repo: str, token: str) -> bool:
    """Can this token actually pull the weights, licence accepted and all?"""
    try:
        r = requests.head(
            FILE_URL.format(repo=repo, filename=PROBE_FILE),
            headers=_headers(token),
            timeout=TIMEOUT,
            allow_redirects=False,
        )
    except requests.RequestException:
        return True     # a network blip is not proof the licence is unaccepted
    if r.status_code in (401, 403):
        return False
    return True         # 200, or a redirect to the CDN — either way it is readable


def test_token(token: str) -> str:
    """Confirm the token works and can reach every gated pyannote model."""
    token = (token or "").strip()
    if not token:
        raise HFError("Enter a HuggingFace token first.")
    try:
        r = requests.get(WHOAMI_URL, headers=_headers(token), timeout=TIMEOUT)
    except requests.RequestException as e:
        raise HFError(f"Could not reach HuggingFace: {e}") from e
    if r.status_code in (401, 403):
        raise HFError("HuggingFace rejected the token. Check it was copied whole.")
    if r.status_code != 200:
        raise HFError(f"HuggingFace returned {r.status_code}: {r.text[:120]}")
    who = r.json().get("name") or "your account"

    blocked = [repo for repo in GATED_REPOS if not _repo_accessible(repo, token)]
    if blocked:
        listed = "\n".join(f"  • huggingface.co/{repo}" for repo in blocked)
        raise HFError(
            f"Token works ({who}), but {who} has not been granted access to:\n{listed}\n"
            "Open each page while signed in, click \"Agree and access repository\", "
            "then test again."
        )
    return f"HuggingFace token OK — {who}, all pyannote models accessible."
