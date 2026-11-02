"""
All database reads and writes go through these functions.

Two rules apply throughout this file:
  * every value from a user goes in as a ? parameter, never formatted
    into the SQL string;
  * the few places that do format SQL (table and column names) only use
    names from fixed lists defined in this file.
"""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from database.database import get_db

# Participant IDs run PASV1001, PASV1002, ... so they're easy to read out
# and to quote in a withdrawal request. They're still anonymous: the number
# only reflects the order people consented in, nothing about who they are.
PARTICIPANT_PREFIX = "PASV"
PARTICIPANT_FIRST_NUMBER = 1001
_SCAN_TABLES = {"url": "url_scans", "email": "email_scans"}


def now() -> str:
    # Everything is stored in UTC so timestamps compare correctly regardless
    # of clock changes or where the server happens to run.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str, nbytes: int = 5) -> str:
    """Random, unguessable IDs such as P-1A2B3C4D. Nothing in them can identify a person."""
    return f"{prefix}-{secrets.token_hex(nbytes).upper()}"


# ---------------------------------------------------------------------------
# Visitor sessions
# ---------------------------------------------------------------------------

def create_visitor_session() -> tuple[str, str]:
    session_id, visitor_id = new_id("S", 8), new_id("V", 8)
    timestamp = now()
    db = get_db()
    db.execute(
        "INSERT INTO visitor_sessions (session_id, visitor_id, first_access, last_activity, session_status) "
        "VALUES (?, ?, ?, ?, 'active')",
        (session_id, visitor_id, timestamp, timestamp),
    )
    db.commit()
    return session_id, visitor_id


def touch_visitor_session(session_id: str, status: str | None = None) -> bool:
    """Update last-activity (and optionally status). False if the row no longer exists."""
    db = get_db()
    if status:
        cursor = db.execute(
            "UPDATE visitor_sessions SET last_activity = ?, session_status = ? WHERE session_id = ?",
            (now(), status, session_id))
    else:
        cursor = db.execute("UPDATE visitor_sessions SET last_activity = ? WHERE session_id = ?",
                            (now(), session_id))
    db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

def create_participant() -> str:
    """Issue the next sequential ID (PASV1001, PASV1002, ...) and record the consent."""
    db = get_db()
    timestamp = now()
    try:
        # Both inserts happen in one transaction: two people consenting at
        # the same moment can't get the same number, and a failure can't
        # leave a number issued with no participant behind it.
        cursor = db.execute("INSERT INTO participant_sequence (issued_at) VALUES (?)", (timestamp,))
        participant_id = f"{PARTICIPANT_PREFIX}{PARTICIPANT_FIRST_NUMBER - 1 + cursor.lastrowid}"
        db.execute(
            "INSERT INTO participants (participant_id, consent_status, consent_timestamp, research_status) "
            "VALUES (?, 'given', ?, 'active')",
            (participant_id, timestamp))
        db.commit()
    except sqlite3.Error:
        db.rollback()
        raise
    return participant_id


def get_participant(participant_id: str):
    return get_db().execute("SELECT * FROM participants WHERE participant_id = ?", (participant_id,)).fetchone()


def complete_participant(participant_id: str) -> None:
    db = get_db()
    # COALESCE keeps the first completion time if someone somehow finishes twice.
    db.execute(
        "UPDATE participants SET research_status = 'completed', "
        "completion_timestamp = COALESCE(completion_timestamp, ?) WHERE participant_id = ?",
        (now(), participant_id))
    db.commit()


def delete_participant(participant_id: str) -> bool:
    """Used for withdrawal requests. The foreign-key cascades remove their scans too."""
    db = get_db()
    cursor = db.execute("DELETE FROM participants WHERE participant_id = ?", (participant_id,))
    db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------

def record_scan(kind: str, scan_id: str, participant_id: str, prediction: str, model_version: str) -> None:
    """Store a completed scan for the statistics. The URL/email text itself is not stored."""
    table = _SCAN_TABLES[kind]
    db = get_db()
    db.execute(
        f"INSERT INTO {table} (scan_id, participant_id, scan_timestamp, prediction_category, model_version, scan_status) "
        "VALUES (?, ?, ?, ?, ?, 'completed')",
        (scan_id, participant_id, now(), prediction, model_version))
    db.commit()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _count(sql: str, params: tuple = ()) -> int:
    return get_db().execute(sql, params).fetchone()[0]


def public_stats() -> dict:
    """The four numbers on the participant dashboard. Totals only, nothing per person."""
    url_scans = _count("SELECT COUNT(*) FROM url_scans WHERE scan_status = 'completed'")
    email_scans = _count("SELECT COUNT(*) FROM email_scans WHERE scan_status = 'completed'")
    return {
        "visitors": _count("SELECT COUNT(*) FROM visitor_sessions"),
        "url_scans": url_scans,
        "email_scans": email_scans,
        "total_scans": url_scans + email_scans,
    }


def admin_stats() -> dict:
    stats = public_stats()
    stats.update({
        "consents": _count("SELECT COUNT(*) FROM participants WHERE consent_status = 'given'"),
        "completed_sessions": _count("SELECT COUNT(*) FROM participants WHERE research_status = 'completed'"),
        "declined_sessions": _count("SELECT COUNT(*) FROM visitor_sessions WHERE session_status = 'declined'"),
        "url_phishing": _count("SELECT COUNT(*) FROM url_scans WHERE prediction_category = 'Phishing'"),
        "email_phishing": _count("SELECT COUNT(*) FROM email_scans WHERE prediction_category = 'Phishing'"),
    })
    return stats


def participant_activity(id_filter: str | None = None, limit: int = 500) -> list:
    """One row per participant for the admin activity table."""
    sql = """
        SELECT p.participant_id, p.consent_status, p.consent_timestamp, p.research_status, p.completion_timestamp,
               (SELECT COUNT(*) FROM url_scans u   WHERE u.participant_id = p.participant_id) AS url_scans,
               (SELECT COUNT(*) FROM email_scans e WHERE e.participant_id = p.participant_id) AS email_scans
        FROM participants p
    """
    params: list = []
    if id_filter:
        sql += " WHERE p.participant_id LIKE ?"
        params.append(f"%{id_filter}%")
    sql += " ORDER BY p.consent_timestamp DESC LIMIT ?"
    params.append(limit)
    return get_db().execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Data retention
# ---------------------------------------------------------------------------

def purge_older_than(days: int) -> int:
    """Delete participants (and their cascaded data) and visitor sessions older than `days`."""
    if days <= 0:
        return 0   # 0 means "no retention limit configured", never "delete everything"
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    db = get_db()
    removed = db.execute("DELETE FROM participants WHERE consent_timestamp < ?", (cutoff,)).rowcount
    db.execute("DELETE FROM visitor_sessions WHERE first_access < ?", (cutoff,))
    db.commit()
    return removed


# ---------------------------------------------------------------------------
# Admin accounts
# ---------------------------------------------------------------------------

def get_admin(username: str):
    return get_db().execute("SELECT * FROM admin_users WHERE username = ?", (username,)).fetchone()


def create_admin(username: str, password_hash: str) -> None:
    db = get_db()
    db.execute("INSERT INTO admin_users (username, password_hash, created_at) VALUES (?, ?, ?)",
               (username, password_hash, now()))
    db.commit()


def record_admin_login(admin_id: int) -> None:
    db = get_db()
    db.execute("UPDATE admin_users SET last_login = ? WHERE admin_id = ?", (now(), admin_id))
    db.commit()
