"""
SQLite setup: the schema and a per-request connection.

SQLite fits a single-machine research prototype well: no server to install,
and the whole dataset is one file that can be backed up or handed over with
the dissertation. The file lives in instance/, which isn't a static folder,
so it can't be downloaded through the web server.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import current_app, g

# Submitted URL strings and email bodies are deliberately *not* stored
# anywhere in this schema; only what the statistics need.
# CREATE ... IF NOT EXISTS makes the schema safe to apply on every startup.
SCHEMA = """
CREATE TABLE IF NOT EXISTS visitor_sessions (
    session_id        TEXT PRIMARY KEY,
    visitor_id        TEXT NOT NULL,
    first_access      TEXT NOT NULL,
    last_activity     TEXT NOT NULL,
    session_status    TEXT NOT NULL DEFAULT 'active'      -- active | consented | declined | completed
);

-- Hands out participant numbers (PASV1001, PASV1002, ...). AUTOINCREMENT
-- guarantees a number is never reused, even after a participant is deleted
-- on withdrawal, so an ID in an old export always means the same person.
CREATE TABLE IF NOT EXISTS participant_sequence (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    participant_id        TEXT PRIMARY KEY,
    consent_status        TEXT NOT NULL,                  -- 'given' (rows only exist after consent)
    consent_timestamp     TEXT NOT NULL,
    research_status       TEXT NOT NULL DEFAULT 'active', -- active | completed
    completion_timestamp  TEXT
);

-- ON DELETE CASCADE means a withdrawal request (deleting the participant)
-- also removes all of their scan records in one go.
CREATE TABLE IF NOT EXISTS url_scans (
    scan_id              TEXT PRIMARY KEY,
    participant_id       TEXT NOT NULL REFERENCES participants(participant_id) ON DELETE CASCADE,
    scan_timestamp       TEXT NOT NULL,
    prediction_category  TEXT NOT NULL,                   -- Phishing | Legitimate
    model_version        TEXT NOT NULL,
    scan_status          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS email_scans (
    scan_id              TEXT PRIMARY KEY,
    participant_id       TEXT NOT NULL REFERENCES participants(participant_id) ON DELETE CASCADE,
    scan_timestamp       TEXT NOT NULL,
    prediction_category  TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    scan_status          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_users (
    admin_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username             TEXT NOT NULL UNIQUE,
    password_hash        TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    last_login           TEXT
);

CREATE INDEX IF NOT EXISTS idx_url_scans_participant ON url_scans(participant_id);
CREATE INDEX IF NOT EXISTS idx_email_scans_participant ON email_scans(participant_id);
"""


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row            # rows behave like dicts in the templates
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores the cascades without this
    return conn


def get_db() -> sqlite3.Connection:
    """One connection per request, kept on flask.g and closed in teardown."""
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE_PATH"])
    return g.db


def close_db(_error=None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db(path: str) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def init_app(app) -> None:
    init_db(app.config["DATABASE_PATH"])
    app.teardown_appcontext(close_db)
