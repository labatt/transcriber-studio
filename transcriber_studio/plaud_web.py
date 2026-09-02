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
from collections.abc import Callable
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
    issued_at: float | None = None  # unix seconds, or None when unreadable
    is_refresh_token: bool = False  # typ WRT: mints tokens, is not one
    workspace_id: str = ""          # the wid claim, needed to refresh against

    @property
    def kind(self) -> str:
        """What to call this token in a message to the user."""
        if self.is_refresh_token:
            return "refresh token"
        return "workspace token" if self.is_workspace_token else "user token"

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    @property
    def days_left(self) -> int | None:
        if self.expires_at is None:
            return None
        return max(0, int((self.expires_at - time.time()) // 86400))

    @property
    def lifetime_seconds(self) -> float | None:
        """How long Plaud minted this token for, not how much is left."""
        if self.expires_at is None or self.issued_at is None:
            return None
        return self.expires_at - self.issued_at

    def describe_remaining(self) -> str:
        """Plain words for how much time is left, for the Settings readout."""
        if self.expires_at is None:
            return "unknown"
        left = self.expires_at - time.time()
        if left <= 0:
            return "expired"
        if left < 3600:
            return f"{int(left // 60)} minutes"
        if left < 86400:
            return f"{int(left // 3600)} hours"
        return f"{int(left // 86400)} days"


def _decode_segment(segment: str) -> dict:
    """One base64url JWT segment as a dict, or {} if it will not decode."""
    segment += "=" * (-len(segment) % 4)      # JWTs drop base64 padding
    try:
        return json.loads(base64.urlsafe_b64decode(segment))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return {}


def _decode_header(token: str) -> dict:
    """The first segment of a JWT, or {} if it will not decode."""
    parts = token.split(".")
    return _decode_segment(parts[0]) if len(parts) == 3 else {}


def _decode_payload(token: str) -> dict:
    """The middle segment of a JWT, or {} if it will not decode."""
    parts = token.split(".")
    return _decode_segment(parts[1]) if len(parts) == 3 else {}


def inspect_token(token: str) -> TokenInfo:
    """What kind of token this is and how long it has left.

    Plaud stamps the kind in the JWT's own ``typ`` header, which is the only
    part of this that is unambiguous::

        UT   user token       the pld_ut cookie
        WT   workspace token  the Authorization header in the network tab
        WRT  refresh token    used to mint the two above

    All three are JWTs the API will take, so the header is what tells them
    apart. Claim names are the fallback, and only ``wid`` is reliable there:
    an earlier version of this looked for ``ut_ref``, which no Plaud token has
    ever carried — the claim pointing back at the parent session is ``ut_sid``.

    Lifetimes, read off a real session's ``pld_sessionMeta``: the user and
    workspace tokens last 24 hours and the refresh token 30 days. Nothing here
    lasts a year.

    None of the three is a mistake to accept. The web app puts the *workspace*
    token on the Authorization header for these calls — confirmed against a
    live session — so WT works directly, and WRT is better still because
    :func:`refresh_workspace_token` spends it on a fresh WT for 30 days. A WRT
    with no ``wid`` is the one thing refused: there is nothing to refresh
    against.
    """
    value = normalize_token(token)
    payload = _decode_payload(value)
    expires = payload.get("exp")
    issued = payload.get("iat")
    typ = str(_decode_header(value).get("typ") or "").upper()
    return TokenInfo(
        expires_at=float(expires) if isinstance(expires, (int, float)) else None,
        issued_at=float(issued) if isinstance(issued, (int, float)) else None,
        # typ first; wid catches a workspace token whose header is missing.
        is_workspace_token=typ in ("WT", "WRT") or (not typ and bool(payload.get("wid"))),
        is_refresh_token=typ == "WRT",
        workspace_id=str(payload.get("wid") or ""),
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
    if info.expired:
        raise TokenRejected("That token has already expired. Copy a fresh one.")
    if info.is_refresh_token and not info.workspace_id:
        raise TokenRejected(
            "That refresh token carries no workspace id, so there is nothing "
            "to refresh against. Copy the refreshToken from the workspaceList "
            "entry rather than from anywhere else."
        )
    return info


#: Plaud's wording when a token is well-formed but its session is gone —
#: "invalid or session expired, re-exchange required". Signing out of
#: web.plaud.ai does this to every token that session issued, so it says
#: "copy a fresh one", not "something went wrong".
DEAD_SESSION_HINTS = ("re-exchange", "session expired", "session has expired")


def _check_body(data: dict, action: str = "the change") -> dict:
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
    message = str(data.get("msg") or data.get("message") or "")
    # The status number is kept in the text: it is the only handle on an
    # undocumented API when a message turns out not to mean what it says.
    detail = f"{message} (status {status})" if message else f"status {status}"
    if any(hint in message.lower() for hint in DEAD_SESSION_HINTS):
        raise TokenRejected(
            f"That token's Plaud session has ended — {detail}.\n\n"
            "The token itself is intact; the session it came from is not. "
            "Signing out of web.plaud.ai invalidates every token that session "
            "issued, including one saved here earlier.\n\n"
            "Sign in at web.plaud.ai, then copy the refreshToken that is there "
            "now — Local Storage → the key ending in :workspaceList. Copy it "
            "before signing out again, or it will be dead too."
        )
    raise PlaudWebError(f"Plaud refused {action}: {detail}")


@dataclass(frozen=True)
class WorkspaceCredentials:
    """What one refresh returns: a token to use, and the means to do it again."""

    workspace_token: str
    expires_at: float                   # unix seconds
    refresh_token: str
    refresh_expires_at: float | None    # unix seconds, None when unreported

    @property
    def expired(self) -> bool:
        return self.expires_at <= time.time()

    @property
    def stale(self) -> bool:
        """True with under a minute left, so a slow rename does not race it."""
        return self.expires_at - time.time() < 60


def refresh_workspace_token(
    refresh_token: str,
    api_base: str = DEFAULT_HOST,
    workspace_id: str = "",
    timeout: int = TIMEOUT,
    _redirected: bool = False,
) -> WorkspaceCredentials:
    """Trade the 30-day refresh token for a fresh 24-hour workspace token.

    ``POST {api}/user-app/auth/workspace/refresh/{wid}`` with an empty body and
    the refresh token as the bearer. The workspace id is not asked of the user:
    it is the ``wid`` claim inside the refresh token itself.

    The reply nests under ``data`` and names the token ``workspace_token``, with
    ``access_token`` as an older alias — both are read, because the web app
    reads both. A ``refresh_token`` comes back only when Plaud rotates it, so an
    absent one means keep the current one rather than treat it as a failure.
    """
    token = normalize_token(refresh_token)
    if not token:
        raise TokenRejected(
            "No Plaud refresh token saved. Add one in Settings → Plaud rename."
        )
    wid = workspace_id or inspect_token(token).workspace_id
    if not wid:
        raise TokenRejected("That refresh token carries no workspace id.")

    base = (api_base or DEFAULT_HOST).rstrip("/")
    url = f"{base}/user-app/auth/workspace/refresh/{wid}"
    try:
        r = requests.post(
            url,
            json={},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "Origin": "https://web.plaud.ai",
                "Referer": "https://web.plaud.ai/",
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise PlaudWebError(f"Could not reach Plaud: {e}") from e

    if r.status_code in (401, 403):
        raise TokenRejected(
            "Plaud no longer accepts the saved refresh token — these last "
            "30 days.\n\nCopy a fresh refreshToken into Settings → Plaud rename."
        )
    if not r.ok:
        raise PlaudWebError(f"Plaud returned HTTP {r.status_code} refreshing the token.")
    try:
        body = r.json()
    except ValueError as e:
        raise PlaudWebError("Plaud returned something that was not JSON.") from e
    if not isinstance(body, dict):
        raise PlaudWebError("Plaud returned an unexpected response shape.")

    # The web app follows one redirect to the account's real region and no
    # more, so a server that keeps redirecting cannot spin this forever.
    if body.get("status") == REGION_MISMATCH_STATUS and not _redirected:
        payload = body.get("data") or {}
        moved = (payload.get("domains") or {}).get("api") or payload.get("domain")
        if moved:
            return refresh_workspace_token(
                token, moved, wid, timeout=timeout, _redirected=True
            )

    data = _check_body(body, "the token refresh").get("data") or {}
    minted = data.get("workspace_token") or data.get("access_token") or ""
    if not minted:
        raise PlaudWebError("Plaud's refresh reply carried no workspace token.")
    expires_in = data.get("expires_in")
    refresh_expires_in = data.get("refresh_expires_in")
    now = time.time()
    return WorkspaceCredentials(
        workspace_token=minted,
        # No expires_in means treat it as good for one day, the observed life.
        expires_at=now + float(expires_in if expires_in else 86400),
        # Absent means unrotated, so the token we just used is still current.
        refresh_token=data.get("refresh_token") or token,
        refresh_expires_at=now + float(refresh_expires_in) if refresh_expires_in else None,
    )


class PlaudWebClient:
    """The one write this app makes to Plaud, and the check that it worked."""

    def __init__(self, token: str, api_base: str = DEFAULT_HOST, timeout: int = TIMEOUT):
        self.api_base = (api_base or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
        value = normalize_token(token)
        info = inspect_token(value)
        # A refresh token is not usable as a bearer, so it is kept aside and
        # spent on minting one. Anything else is already the token to send.
        self._refresh_token = value if info.is_refresh_token else ""
        self.token = "" if info.is_refresh_token else value
        self._minted: WorkspaceCredentials | None = None
        self.on_refresh: Callable[[WorkspaceCredentials], None] | None = None

    def _ensure_token(self) -> None:
        """Mint a workspace token if all we hold is the means to make one."""
        if not self._refresh_token:
            return
        if self._minted is not None and not self._minted.stale:
            return
        creds = refresh_workspace_token(
            self._refresh_token, self.api_base, timeout=self.timeout
        )
        self._minted = creds
        self.token = creds.workspace_token
        # Plaud rotates the refresh token sometimes; keep using the current one.
        self._refresh_token = creds.refresh_token
        if self.on_refresh is not None:
            self.on_refresh(creds)

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
        self._ensure_token()
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
                "Plaud no longer accepts the saved token.\n\nA workspace token "
                "lasts 24 hours; the refreshToken lasts 30 days and is spent "
                "on new ones automatically. Copy a fresh one into "
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
        """Prove the token works, without changing anything. Raises on failure.

        A refresh token proves itself: minting is a real authenticated call to
        a real endpoint, and Plaud will not hand back a workspace token for a
        credential it does not accept. Nothing further is worth asking.

        A pasted workspace token has no such call behind it, so it has to be
        spent on something. There is no known read-only endpoint to spend it
        on — ``/team-app/workspaces/list`` was a guess and answers status -1,
        "invalid request" — so this reports honestly that it cannot tell,
        rather than inventing a pass or failing a token that may be fine.
        """
        if self._refresh_token:
            self._ensure_token()
            return
        raise PlaudWebError(
            "A workspace token cannot be checked without spending it on a "
            "rename, so this cannot say whether it works — only a rename will "
            "tell you.\n\nPaste the refreshToken instead and this can verify "
            "it properly: it lasts 30 days, and checking it is a real call."
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
