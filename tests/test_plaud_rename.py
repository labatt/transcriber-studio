# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renaming a Plaud recording, and pushing that name to Plaud.

The push goes through the web app's API rather than the official one, because
the official one is read-only. Two failure modes matter more than the rest:
Plaud reports its own errors in a 200 response body, and it answers the wrong
regional host with a 200 as well. Both would otherwise read as success.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from transcriber_studio import plaud_web


def _jwt(payload: dict, typ: str | None = None) -> str:
    def part(data: dict) -> str:
        raw = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
        return raw.rstrip("=")      # JWTs drop base64 padding
    header = {"alg": "HS256"}
    if typ is not None:
        header["typ"] = typ
    return f"{part(header)}.{part(payload)}.signature"


def _user_token(hours: int = 24) -> str:
    """A real pld_ut: typ UT, and minted for a day rather than a year."""
    now = time.time()
    return _jwt(
        {"exp": now + hours * 3600, "iat": now, "sub": "u1", "sid": "s1"}, typ="UT"
    )


# ---- token handling --------------------------------------------------
def test_a_pasted_bearer_prefix_is_accepted():
    assert plaud_web.normalize_token("  Bearer abc.def.ghi ") == "abc.def.ghi"
    assert plaud_web.normalize_token('"abc.def.ghi"') == "abc.def.ghi"


def test_a_user_token_passes_validation():
    info = plaud_web.validate_token(_user_token())
    assert not info.is_workspace_token
    assert not info.expired


def test_a_workspace_token_is_accepted_because_it_is_the_one_that_works():
    """The web app puts the WT on its own Authorization header for this API.

    An earlier version refused it and sent the user after the pld_ut cookie,
    which is HttpOnly and never reaches these calls at all.
    """
    token = _jwt({"exp": time.time() + 86400, "wid": "ws_1", "ut_sid": "s1"}, typ="WT")
    info = plaud_web.validate_token(token)
    assert info.is_workspace_token
    assert not info.is_refresh_token
    assert info.kind == "workspace token"


def test_a_refresh_token_is_accepted_and_carries_its_workspace():
    """WRT is the one worth pasting: 30 days, and the client mints from it.

    The workspace id is never asked of the user — it is the wid claim, which
    is what /user-app/auth/workspace/refresh/{wid} needs.
    """
    token = _jwt({"exp": time.time() + 30 * 86400, "wid": "ws_1"}, typ="WRT")
    info = plaud_web.validate_token(token)
    assert info.is_refresh_token
    assert info.workspace_id == "ws_1"
    assert info.kind == "refresh token"


def test_a_refresh_token_without_a_workspace_is_refused():
    """There would be nothing to refresh against."""
    token = _jwt({"exp": time.time() + 30 * 86400}, typ="WRT")
    with pytest.raises(plaud_web.TokenRejected, match="workspace id"):
        plaud_web.validate_token(token)


def test_a_workspace_token_without_a_type_header_is_still_recognised():
    """The wid claim is the fallback when the header says nothing."""
    info = plaud_web.inspect_token(_jwt({"exp": time.time() + 86400, "wid": "ws_1"}))
    assert info.is_workspace_token
    assert not info.is_refresh_token


def test_ut_sid_alone_does_not_make_a_token_a_workspace_token():
    """A user token carries a session id; only wid means workspace-scoped.

    The previous check looked for a claim named ut_ref, which no Plaud token
    has ever carried, so that half of it never fired.
    """
    assert not plaud_web.inspect_token(
        _jwt({"exp": time.time() + 86400, "ut_sid": "s1"})
    ).is_workspace_token
    assert not plaud_web.inspect_token(
        _jwt({"exp": time.time() + 86400, "ut_ref": "s1"})
    ).is_workspace_token


def test_the_user_token_lifetime_is_reported_in_hours_not_days():
    """A 24h token rounds to zero whole days, which reads as 'already dead'."""
    info = plaud_web.validate_token(_user_token(hours=24))
    assert info.lifetime_seconds == pytest.approx(86400, abs=5)
    assert info.days_left == 0                      # why days_left is not shown
    assert "hour" in info.describe_remaining()


def test_remaining_time_is_described_at_a_sensible_scale():
    now = time.time()
    def left(seconds):
        return plaud_web.inspect_token(
            _jwt({"exp": now + seconds, "iat": now}, typ="UT")
        ).describe_remaining()

    assert "minutes" in left(600)
    assert "hours" in left(5 * 3600)
    assert "days" in left(10 * 86400)
    assert left(-60) == "expired"


def test_an_expired_token_is_refused():
    with pytest.raises(plaud_web.TokenRejected, match="expired"):
        plaud_web.validate_token(_jwt({"exp": time.time() - 60}))


def test_something_that_is_not_a_token_is_refused():
    with pytest.raises(plaud_web.TokenRejected):
        plaud_web.validate_token("not-a-token")
    with pytest.raises(plaud_web.TokenRejected):
        plaud_web.validate_token("")


def test_an_unreadable_token_is_not_mistaken_for_expired():
    """No exp claim means unknown, which must not read as 'already dead'."""
    info = plaud_web.inspect_token(_jwt({"uid": "u1"}))
    assert info.expires_at is None
    assert not info.expired
    assert info.days_left is None


# ---- the rename call -------------------------------------------------
class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _client(monkeypatch, payload, status_code=200, seen=None):
    def fake_request(method, url, headers=None, json=None, timeout=None):
        if seen is not None:
            seen.update(method=method, url=url, headers=headers, body=json)
        return _Response(payload, status_code)

    monkeypatch.setattr(plaud_web.requests, "request", fake_request)
    return plaud_web.PlaudWebClient(_user_token())


def test_a_successful_rename_sends_only_the_filename(monkeypatch):
    """The endpoint also accepts tag lists and transcription configs.

    Sending anything beyond the name risks moving folders or starting a cloud
    transcription as a side effect of a rename.
    """
    seen = {}
    client = _client(monkeypatch, {"status": 0, "msg": "ok"}, seen=seen)
    client.rename("abc123", "Weekly sync")

    assert seen["method"] == "PATCH"
    assert seen["url"] == "https://api.plaud.ai/file/abc123"
    assert seen["body"] == {"filename": "Weekly sync"}
    assert seen["headers"]["Authorization"].startswith("Bearer ")


def test_an_error_wearing_an_http_200_is_not_success(monkeypatch):
    """Plaud reports its own failures in the body and still answers 200."""
    client = _client(monkeypatch, {"status": -1, "msg": "file not found"})
    with pytest.raises(plaud_web.PlaudWebError, match="file not found"):
        client.rename("abc123", "New name")


def test_the_wrong_region_is_reported_as_the_wrong_region(monkeypatch):
    """Also a 200. Silently ignoring it reports renames that never happened."""
    client = _client(monkeypatch, {
        "status": -302,
        "msg": "user region mismatch",
        "data": {"domains": {"api": "https://api-euc1.plaud.ai"}},
    })
    with pytest.raises(plaud_web.PlaudWebError) as excinfo:
        client.rename("abc123", "New name")
    assert "api-euc1.plaud.ai" in str(excinfo.value)
    assert "different Plaud server" in str(excinfo.value)


def test_an_expired_token_says_so(monkeypatch):
    client = _client(monkeypatch, {}, status_code=401)
    with pytest.raises(plaud_web.TokenRejected, match="no longer accepts"):
        client.rename("abc123", "New name")


def test_a_missing_token_does_not_reach_the_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not have made a request")

    monkeypatch.setattr(plaud_web.requests, "request", explode)
    with pytest.raises(plaud_web.TokenRejected, match="No Plaud web token"):
        plaud_web.PlaudWebClient("").rename("abc123", "New name")


def test_an_empty_name_is_refused_before_the_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not have made a request")

    monkeypatch.setattr(plaud_web.requests, "request", explode)
    with pytest.raises(plaud_web.PlaudWebError, match="needs a name"):
        plaud_web.PlaudWebClient(_user_token()).rename("abc123", "   ")


def test_a_non_json_answer_is_an_error_not_a_crash(monkeypatch):
    client = _client(monkeypatch, ValueError("no json"))
    with pytest.raises(plaud_web.PlaudWebError, match="not JSON"):
        client.rename("abc123", "New name")


def test_the_region_choice_picks_the_host(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        plaud_web.requests, "request",
        lambda method, url, headers=None, json=None, timeout=None: (
            seen.update(url=url), _Response({"status": 0}))[1],
    )
    plaud_web.PlaudWebClient(
        _user_token(), plaud_web.API_HOSTS["eu"]
    ).rename("abc", "n")
    assert seen["url"].startswith("https://api-euc1.plaud.ai/")


# ---- refreshing a workspace token ------------------------------------
def _refresh_token(days: int = 30, wid: str = "ws_1") -> str:
    now = time.time()
    return _jwt(
        {"exp": now + days * 86400, "iat": now, "wid": wid, "ut_sid": "s1"}, typ="WRT"
    )


def _fake_post(monkeypatch, payload, status_code=200, seen=None):
    def fake_post(url, json=None, headers=None, timeout=None):
        if seen is not None:
            seen.update(url=url, body=json, headers=headers)
        return _Response(payload, status_code)

    monkeypatch.setattr(plaud_web.requests, "post", fake_post)


def test_a_refresh_spends_the_refresh_token_on_a_workspace_token(monkeypatch):
    """Empty body, refresh token as the bearer, workspace id from its own wid."""
    seen = {}
    _fake_post(
        monkeypatch,
        {"status": 0, "data": {"workspace_token": "new.wt.token", "expires_in": 86400}},
        seen=seen,
    )
    creds = plaud_web.refresh_workspace_token(_refresh_token(wid="ws_abc"))

    assert seen["url"] == "https://api.plaud.ai/user-app/auth/workspace/refresh/ws_abc"
    assert seen["body"] == {}
    assert seen["headers"]["Authorization"].startswith("Bearer ")
    assert creds.workspace_token == "new.wt.token"
    assert not creds.expired


def test_the_older_access_token_alias_is_read_too(monkeypatch):
    _fake_post(
        monkeypatch,
        {"status": 0, "data": {"access_token": "aliased.wt", "expires_in": 3600}},
    )
    assert plaud_web.refresh_workspace_token(_refresh_token()).workspace_token == "aliased.wt"


def test_an_unrotated_refresh_token_is_kept_not_blanked(monkeypatch):
    """Plaud omits refresh_token when it has not rotated it."""
    original = _refresh_token()
    _fake_post(
        monkeypatch, {"status": 0, "data": {"workspace_token": "wt", "expires_in": 60}}
    )
    creds = plaud_web.refresh_workspace_token(original)
    assert creds.refresh_token == original
    assert creds.refresh_expires_at is None


def test_a_rotated_refresh_token_replaces_the_old_one(monkeypatch):
    _fake_post(
        monkeypatch,
        {
            "status": 0,
            "data": {
                "workspace_token": "wt",
                "expires_in": 86400,
                "refresh_token": "rotated.wrt",
                "refresh_expires_in": 2592000,
            },
        },
    )
    creds = plaud_web.refresh_workspace_token(_refresh_token())
    assert creds.refresh_token == "rotated.wrt"
    assert creds.refresh_expires_at is not None


def test_a_refresh_follows_one_region_redirect_and_no_more(monkeypatch):
    """-302 carries the right host; the web app retries once, so this does too."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return _Response(
                {"status": -302, "data": {"domains": {"api": "https://api-euc1.plaud.ai"}}}
            )
        return _Response({"status": 0, "data": {"workspace_token": "wt", "expires_in": 60}})

    monkeypatch.setattr(plaud_web.requests, "post", fake_post)
    creds = plaud_web.refresh_workspace_token(_refresh_token(wid="ws_1"))

    assert creds.workspace_token == "wt"
    assert calls == [
        "https://api.plaud.ai/user-app/auth/workspace/refresh/ws_1",
        "https://api-euc1.plaud.ai/user-app/auth/workspace/refresh/ws_1",
    ]


def test_an_endlessly_redirecting_server_does_not_loop(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return _Response({"status": -302, "data": {"domains": {"api": "https://api.plaud.ai"}}})

    monkeypatch.setattr(plaud_web.requests, "post", fake_post)
    with pytest.raises(plaud_web.PlaudWebError):
        plaud_web.refresh_workspace_token(_refresh_token())
    assert len(calls) == 2          # the original and one redirect, then stop


def test_a_rejected_refresh_token_says_to_replace_it(monkeypatch):
    _fake_post(monkeypatch, {}, status_code=401)
    with pytest.raises(plaud_web.TokenRejected, match="30 days"):
        plaud_web.refresh_workspace_token(_refresh_token())


def test_a_refresh_reply_without_a_token_is_an_error_not_an_empty_bearer(monkeypatch):
    _fake_post(monkeypatch, {"status": 0, "data": {"expires_in": 86400}})
    with pytest.raises(plaud_web.PlaudWebError, match="no workspace token"):
        plaud_web.refresh_workspace_token(_refresh_token())


def test_a_client_given_a_refresh_token_mints_one_before_calling(monkeypatch):
    """The refresh token is never sent as the bearer on a real call."""
    seen = {}
    _fake_post(
        monkeypatch,
        {"status": 0, "data": {"workspace_token": "minted.wt", "expires_in": 86400}},
    )

    def fake_request(method, url, headers=None, json=None, timeout=None):
        seen.update(headers=headers)
        return _Response({"status": 0})

    monkeypatch.setattr(plaud_web.requests, "request", fake_request)
    client = plaud_web.PlaudWebClient(_refresh_token())
    client.rename("abc123", "Weekly sync")

    assert seen["headers"]["Authorization"] == "Bearer minted.wt"


def test_a_client_mints_once_and_reuses_it(monkeypatch):
    mints = []

    def fake_post(url, json=None, headers=None, timeout=None):
        mints.append(url)
        return _Response(
            {"status": 0, "data": {"workspace_token": "minted.wt", "expires_in": 86400}}
        )

    monkeypatch.setattr(plaud_web.requests, "post", fake_post)
    monkeypatch.setattr(
        plaud_web.requests, "request",
        lambda *a, **k: _Response({"status": 0}),
    )
    client = plaud_web.PlaudWebClient(_refresh_token())
    client.rename("a", "one")
    client.rename("b", "two")
    assert len(mints) == 1


def test_a_plain_workspace_token_is_used_as_is_without_refreshing(monkeypatch):
    """Nothing should call the refresh endpoint when a usable token was pasted."""
    def explode(*a, **k):
        raise AssertionError("should not refresh when given a usable token")

    monkeypatch.setattr(plaud_web.requests, "post", explode)
    seen = {}
    client = _client(monkeypatch, {"status": 0}, seen=seen)
    client.rename("abc123", "Weekly sync")
    assert seen["headers"]["Authorization"].startswith("Bearer ")


def test_a_dead_session_says_to_copy_a_fresh_token(monkeypatch):
    """Plaud's "re-exchange required" means the session ended, not a bad token.

    Signing out of web.plaud.ai kills every token that session issued, so this
    has to read as "copy the current one", not as a generic refusal.
    """
    _fake_post(
        monkeypatch,
        {"status": -1001, "msg": "invalid or session expired, re-exchange required"},
    )
    with pytest.raises(plaud_web.TokenRejected) as excinfo:
        plaud_web.refresh_workspace_token(_refresh_token())
    text = str(excinfo.value)
    assert "session has ended" in text
    assert "workspaceList" in text
    assert "status -1001" in text          # the code stays visible for diagnosis


def test_an_ordinary_refusal_names_what_was_refused(monkeypatch):
    _fake_post(monkeypatch, {"status": -7, "msg": "nope"})
    with pytest.raises(plaud_web.PlaudWebError, match="refused the token refresh"):
        plaud_web.refresh_workspace_token(_refresh_token())


def test_a_rename_refusal_still_names_the_change(monkeypatch):
    client = _client(monkeypatch, {"status": -7, "msg": "nope"})
    with pytest.raises(plaud_web.PlaudWebError, match="refused the change"):
        client.rename("abc123", "New name")
