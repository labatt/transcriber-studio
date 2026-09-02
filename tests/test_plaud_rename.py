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


def _jwt(payload: dict) -> str:
    def part(data: dict) -> str:
        raw = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
        return raw.rstrip("=")      # JWTs drop base64 padding
    return f"{part({'alg': 'HS256'})}.{part(payload)}.signature"


def _user_token(days: int = 300) -> str:
    return _jwt({"exp": time.time() + days * 86400, "uid": "u1"})


# ---- token handling --------------------------------------------------
def test_a_pasted_bearer_prefix_is_accepted():
    assert plaud_web.normalize_token("  Bearer abc.def.ghi ") == "abc.def.ghi"
    assert plaud_web.normalize_token('"abc.def.ghi"') == "abc.def.ghi"


def test_a_user_token_passes_validation():
    info = plaud_web.validate_token(_user_token())
    assert not info.is_workspace_token
    assert info.days_left is not None and info.days_left > 200


@pytest.mark.parametrize("claim", ["ut_ref", "wid"])
def test_a_workspace_token_is_refused(claim):
    """The one on the Authorization header lasts a day, and is easy to grab."""
    token = _jwt({"exp": time.time() + 86400, claim: "w1"})
    with pytest.raises(plaud_web.TokenRejected, match="workspace token"):
        plaud_web.validate_token(token)


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
