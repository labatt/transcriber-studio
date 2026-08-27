# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
"""SQLite store for per-model AI parameters learned from successful calls."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import APP_DIR

DB_PATH = APP_DIR / "ai_models.db"


@dataclass
class ModelProfile:
    provider: str
    model_id: str
    max_tokens: int = 8192
    temperature: float = 0.2
    use_max_completion_tokens: bool = False
    omit_temperature: bool = False
    omit_json_response_format: bool = False

    def with_changes(self, **kwargs) -> ModelProfile:
        data = {
            "provider": self.provider,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_max_completion_tokens": self.use_max_completion_tokens,
            "omit_temperature": self.omit_temperature,
            "omit_json_response_format": self.omit_json_response_format,
        }
        data.update(kwargs)
        return ModelProfile(**data)


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(model_profile)")}
    if "omit_json_response_format" not in cols:
        conn.execute(
            "ALTER TABLE model_profile ADD COLUMN omit_json_response_format INTEGER NOT NULL DEFAULT 0"
        )


def _connect() -> sqlite3.Connection:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_profile (
            provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            max_tokens INTEGER NOT NULL DEFAULT 4096,
            temperature REAL NOT NULL DEFAULT 0.2,
            use_max_completion_tokens INTEGER NOT NULL DEFAULT 0,
            omit_temperature INTEGER NOT NULL DEFAULT 0,
            omit_json_response_format INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (provider, model_id)
        )
        """
    )
    _migrate(conn)
    return conn


def load_profile(provider: str, model_id: str) -> ModelProfile:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM model_profile WHERE provider = ? AND model_id = ?",
            (provider, model_id),
        ).fetchone()
    if not row:
        return ModelProfile(provider=provider, model_id=model_id)
    return ModelProfile(
        provider=row["provider"],
        model_id=row["model_id"],
        max_tokens=int(row["max_tokens"]),
        temperature=float(row["temperature"]),
        use_max_completion_tokens=bool(row["use_max_completion_tokens"]),
        omit_temperature=bool(row["omit_temperature"]),
        omit_json_response_format=bool(row["omit_json_response_format"]),
    )


def save_profile(profile: ModelProfile) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO model_profile (
                provider, model_id, max_tokens, temperature,
                use_max_completion_tokens, omit_temperature,
                omit_json_response_format, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model_id) DO UPDATE SET
                max_tokens = excluded.max_tokens,
                temperature = excluded.temperature,
                use_max_completion_tokens = excluded.use_max_completion_tokens,
                omit_temperature = excluded.omit_temperature,
                omit_json_response_format = excluded.omit_json_response_format,
                updated_at = excluded.updated_at
            """,
            (
                profile.provider,
                profile.model_id,
                profile.max_tokens,
                profile.temperature,
                int(profile.use_max_completion_tokens),
                int(profile.omit_temperature),
                int(profile.omit_json_response_format),
                now,
            ),
        )
        conn.commit()


def suggest_profile_fix(
    error: str, profile: ModelProfile, *, provider: str = ""
) -> ModelProfile | None:
    """Return an adjusted profile when an API error looks parameter-related."""
    msg = error.lower()
    uses_json_mode = provider in {"openai", "openrouter", "grok"}
    param_rejected = (
        "unsupported",
        "not support",
        "does not support",
        "invalid",
        "deprecated",
        "not allowed",
        "must not",
        "cannot use",
        "unrecognized",
        "unknown parameter",
    )
    if (
        ("max_tokens" in msg or "max_completion_tokens" in msg)
        and not profile.use_max_completion_tokens
        and any(k in msg for k in param_rejected)
    ):
        return profile.with_changes(use_max_completion_tokens=True)
    if "temperature" in msg and any(k in msg for k in param_rejected):
        if not profile.omit_temperature:
            return profile.with_changes(omit_temperature=True)
    if uses_json_mode and ("response_format" in msg or "json_object" in msg) and any(
        k in msg for k in param_rejected
    ):
        if not profile.omit_json_response_format:
            return profile.with_changes(omit_json_response_format=True)
    if any(
        k in msg
        for k in (
            "empty response",
            "invalid json",
            "truncated json",
            "expecting value",
            "no json",
            "missing 'segments'",
            "stop_reason=max_tokens",
            "finish_reason=length",
        )
    ):
        if uses_json_mode and not profile.omit_json_response_format:
            return profile.with_changes(omit_json_response_format=True)
        cap = 8192 if provider == "anthropic" else 16384
        if profile.max_tokens < cap:
            return profile.with_changes(max_tokens=min(cap, max(profile.max_tokens * 2, 8192)))
    if "finish_reason=length" in msg.replace(" ", "_") or "max tokens" in msg:
        cap = 8192 if provider == "anthropic" else 16384
        if profile.max_tokens < cap:
            return profile.with_changes(max_tokens=min(cap, profile.max_tokens * 2))
    if "max_tokens" in msg or "max_completion_tokens" in msg or "output tokens" in msg:
        if profile.max_tokens > 1024:
            return profile.with_changes(max_tokens=max(1024, profile.max_tokens // 2))
    if "context" in msg and ("length" in msg or "window" in msg or "too long" in msg):
        if profile.max_tokens > 512:
            return profile.with_changes(max_tokens=max(512, profile.max_tokens // 2))
    return None


def profile_to_json(profile: ModelProfile) -> str:
    return json.dumps(profile.__dict__)
