"""
App settings.

Anything secret or study-specific comes from environment variables, usually
via a local .env file that is git-ignored. Nothing sensitive is hard-coded
here; the defaults are only what's needed to run on a laptop.
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """
    Read KEY=VALUE lines from .env into os.environ.

    python-dotenv is used if it's installed. The fallback parser is there so
    the app still starts on a machine where someone skipped
    `pip install -r requirements.txt`. Real environment variables always win
    over the file.
    """
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Config:
    APP_ENV = os.environ.get("APP_ENV", "development")

    # Without a fixed key every restart logs everyone out, which is annoying
    # mid-study but harmless. app.py logs a warning when this fallback is used.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    # instance/ sits outside static/, so Flask never serves the database file.
    DATABASE_PATH = os.environ.get("DATABASE_PATH", str(ROOT / "instance" / "phisscan.sqlite3"))

    # Session cookie: not readable from JavaScript, not sent on cross-site
    # POSTs, and HTTPS-only once deployed.
    SESSION_COOKIE_NAME = "phisscan_session"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", APP_ENV == "production")
    PERMANENT_SESSION_LIFETIME = timedelta(hours=_env_int("SESSION_LIFETIME_HOURS", 4))
    ADMIN_SESSION_MINUTES = _env_int("ADMIN_SESSION_MINUTES", 30)

    # The largest legitimate request is a 50k-character pasted email.
    # Anything bigger is rejected by Flask before it reaches a view.
    MAX_CONTENT_LENGTH = 256 * 1024

    # Used once, on first start, to create the admin account. Only the hash
    # is stored; the plain value never goes into the database.
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "akik")
    ADMIN_INITIAL_PASSWORD = os.environ.get("ADMIN_INITIAL_PASSWORD", "")
    LOGIN_MAX_ATTEMPTS = _env_int("LOGIN_MAX_ATTEMPTS", 5)
    LOGIN_WINDOW_MINUTES = _env_int("LOGIN_WINDOW_MINUTES", 15)

    # Study details. These have to match the approved Participant Information
    # Sheet, so they're configurable rather than written into the templates.
    # The withdrawal text deliberately has no deadline until one is approved.
    RESEARCHER_NAME = os.environ.get("RESEARCHER_NAME", "Akik Abdullah")
    UNIVERSITY = os.environ.get("UNIVERSITY", "Abertay University")
    RESEARCHER_CONTACT = os.environ.get("RESEARCHER_CONTACT", "")
    SUPERVISOR_CONTACT = os.environ.get("SUPERVISOR_CONTACT", "")
    WITHDRAWAL_INFO = os.environ.get(
        "WITHDRAWAL_INFO",
        "You may stop at any time by closing the browser. To withdraw your data, contact the researcher "
        "quoting your anonymous participant ID. [Withdrawal deadline to be set according to the approved "
        "Participant Information Sheet.]",
    )
    DATA_RETENTION_DAYS = _env_int("DATA_RETENTION_DAYS", 0)   # 0 = keep until removed by hand

    # Scan reports live only in memory and disappear after this long.
    REPORT_CACHE_MINUTES = _env_int("REPORT_CACHE_MINUTES", 60)

    # Only the test suite turns this off.
    CSRF_ENABLED = True
    TESTING = False
