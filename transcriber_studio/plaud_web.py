# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renaming a Plaud recording, through the web app's own API.

This is a second, entirely separate way of talking to Plaud, and it exists for
one reason: the official API cannot rename anything.

The rest of the app goes through the `plaud` CLI, which authenticates by OAuth
and stores its tokens in ~/.plaud/tokens.json. That API is read-only. Every
write verb on a file is refused::

    PATCH /open/third-party/files/{id}  ->  405 Method Not Allowed, Allow: GET

The web app at web.plaud.ai uses a different host and a different API, and that
one does support renaming. It is not documented and not promised to anyone, so
everything here is written to fail safely and say so:

* nothing happens without a token the user has deliberately pasted in;
* the local rename is committed first, so a push that fails costs the name only
  on Plaud's side, never in this app;
* every answer is checked twice — once for the HTTP status, and again for the
  ``status`` field in the body, because this API returns 200 with a non-zero
  status for its own errors and a caller that trusts the HTTP code alone will
  report renames that never happened.

Expect this to break one day. When it does it will break by refusing, not by
corrupting anything: the endpoint takes one field, and the field is the name.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from dataclasses import dataclass

import requests

#: Plaud runs one API host per region and an account lives on exactly one of
#: them. The wrong one does not error — it answers 200 with status -302 and the
#: right host in the body, which is why _check_body looks for that.
API_HOSTS = {
    "global": "https://api.plaud.ai",
    "eu": "https://api-euc1.plaud.ai",
    "apac": "https://api-apse1.plaud.ai",
}
DEFAULT_HOST = API_HOSTS["global"]

#: The web app's own user agent. Sent because this is the web app's API and a
#: bare python-requests string is the kind of thing that gets rate-limited.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: Plaud's own success value in a response body. Anything else is a failure
#: wearing an HTTP 200.
OK_STATUS = 0
#: "You are on the wrong regional host." The body carries the right one.
REGION_MISMATCH_STATUS = -302

TIMEOUT = 30


class PlaudWebError(RuntimeError):
    """Something went wrong that the user can act on."""


class TokenRejected(PlaudWebError):
    """The token is missing, malformed, or no longer accepted."""


@dataclass(frozen=True)
class TokenInfo:
    """What can be read out of a pasted token without asking Plaud."""

    expires_at: float | None        # unix seconds, or None when unreadable
    is_workspace_token: bool

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    @property
    def days_left(self) -> int | None:
        if self.expires_at is None:
            return None
        return max(0, int((self.expires_at - time.time()) // 86400))


def _decode_payload(token: str) -> dict:
    """The middle segment of a JWT, or {} if it will not decode."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    segment = parts[1]
    segment += "=" * (-len(segment) % 4)      # JWTs drop base64 padding
    try:
        return json.loads(base64.urlsafe_b64decode(segment))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return {}


def inspect_token(token: str) -> TokenInfo:
    """What kind of token this is and how long it has left.

    The distinction that matters is workspace token versus user token. Both are
    JWTs, both are accepted by the API, and the one the browser puts on request
    headers — the tempting one to copy out of the network tab — is the
    workspace token, which lasts about a day. The user token lives in the
    ``pld_ut`` cookie and lasts the better part of a year. Pasting the wrong one
    works perfectly until tomorrow, so it is worth refusing up front.
    """
    payload = _decode_payload(normalize_token(token))
    expires = payload.get("exp")
    return TokenInfo(
        expires_at=float(expires) if isinstance(expires, (int, float)) else None,
        # How Plaud marks a workspace-scoped token: a reference back to the user
        # token it was minted from, and the workspace it was minted for.
        is_workspace_token=bool(payload.get("ut_ref") or payload.get("wid")),
    )


def normalize_token(token: str) -> str:
    """Accept a pasted value with or without the ``Bearer`` prefix."""
    value = (token or "").strip().strip('"').strip("'")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def validate_token(token: str) -> TokenInfo:
    """Check a pasted token's shape before it is stored. Raises TokenRejected."""
    value = normalize_token(token)
    if not value:
        raise TokenRejected("Paste the token first.")
    if value.count(".") != 2:
        raise TokenRejected(
            "That does not look like a Plaud token. It should be three "
            "dot-separated blocks of letters and numbers."
        )
    info = inspect_token(value)
    if info.is_workspace_token:
        raise TokenRejected(
            "That is a workspace token, which stops working within a day.\n\n"
            "It is the one on the Authorization header in the network tab — an "
            "easy one to copy by mistake. The one that lasts is the pld_ut "
            "cookie:\n\n"
            "  DevTools → Application → Cookies → https://web.plaud.ai → pld_ut"
        )
    if info.expired:
        raise TokenRejected("That token has already expired. Copy a fresh one.")
    return info


def _check_body(data: dict) -> dict:
    """Raise on a failure that arrived wearing an HTTP 200.

    This API reports its own errors in the body and still answers 200. A caller
    that only looks at the HTTP status reports success for a rename that did not
    happen, which is exactly the failure this app must not have.
    """
    status = data.get("status")
    if status == OK_STATUS:
        return data
    if status == REGION_MISMATCH_STATUS:
        right_host = (data.get("data") or {}).get("domains", {}).get("api", "")
        raise PlaudWebError(
            "This account lives on a different Plaud server"
            + (f" ({right_host})." if right_host else ".")
            + "\n\nChange the server in Settings → Plaud rename and try again."
        )
    message = data.get("msg") or data.get("message") or f"status {status}"
    raise PlaudWebError(f"Plaud refused the change: {message}")


class PlaudWebClient:
    """The one write this app makes to Plaud, and the check that it worked."""

    def __init__(self, token: str, api_base: str = DEFAULT_HOST, timeout: int = TIMEOUT):
        self.token = normalize_token(token)
        self.api_base = (api_base or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Origin": "https://web.plaud.ai",
            "Referer": "https://web.plaud.ai/",
            # Plaud's own attribution for where an edit came from. Harmless if
            # ignored, and honest about the fact that this is the web API.
            "edit-from": "web",
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if not self.token:
            raise TokenRejected(
                "No Plaud web token saved. Add one in Settings → Plaud rename "
                "to let renames reach Plaud."
            )
        url = f"{self.api_base}{path}"
        try:
            r = requests.request(
                method, url, headers=self._headers(), json=body, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise PlaudWebError(f"Could not reach Plaud: {e}") from e
        if r.status_code in (401, 403):
            raise TokenRejected(
                "Plaud no longer accepts the saved token — these expire after "
                "about a year.\n\nCopy a fresh pld_ut cookie into "
                "Settings → Plaud rename."
            )
        if not r.ok:
            raise PlaudWebError(f"Plaud returned HTTP {r.status_code} for {method} {path}.")
        try:
            data = r.json()
        except ValueError as e:
            raise PlaudWebError("Plaud returned something that was not JSON.") from e
        if not isinstance(data, dict):
            raise PlaudWebError("Plaud returned an unexpected response shape.")
        return _check_body(data)

    # ---- the only two calls -------------------------------------------
    def check(self) -> None:
        """Prove the token works, without changing anything. Raises on failure."""
        self._request(
            "GET", "/team-app/workspaces/list?need_personal_workspace=true"
        )

    def rename(self, file_id: str, filename: str) -> None:
        """Push a new name for one recording. Raises on any failure.

        Deliberately sends nothing but the name. The endpoint behind this is a
        general metadata patch: the same call replaces folder membership when
        given ``filetag_id_list`` and starts a cloud transcription when given
        ``tranConfig``. One field in, one field changed.
        """
        name = (filename or "").strip()
        if not name:
            raise PlaudWebError("A recording needs a name.")
        self._request("PATCH", f"/file/{file_id}", {"filename": name})
