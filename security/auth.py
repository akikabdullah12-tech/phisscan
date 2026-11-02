"""
Access control and admin password handling.

Two kinds of protected area:
  * participant pages, open only after consent has been given in this
    browser session (participant_required);
  * admin pages, which need a logged-in admin whose session hasn't been
    idle for too long (admin_required).

Passwords are hashed with Werkzeug's default scheme (scrypt in current
versions, salted). The plain text is never stored or logged.
"""

from __future__ import annotations

import re
import time
from functools import wraps

from flask import current_app, flash, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database import models

# Checking a password against this dummy hash when the username doesn't
# exist means a wrong username and a wrong password take about the same
# time, so response times don't reveal which usernames are real.
_DUMMY_HASH = generate_password_hash("placeholder-used-only-for-timing")


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    matches = check_password_hash(stored_hash or _DUMMY_HASH, password)
    return matches and stored_hash is not None


def password_problems(password: str, username: str = "") -> list[str]:
    """Why a password is too weak for production (an empty list means it's acceptable)."""
    problems = []
    if len(password) < 12:
        problems.append("at least 12 characters")
    if not (re.search(r"[A-Z]", password) and re.search(r"[a-z]", password)):
        problems.append("both upper- and lower-case letters")
    if not re.search(r"\d", password):
        problems.append("at least one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("at least one symbol")
    if username and username.lower() in password.lower():
        problems.append("must not contain the username")
    return problems


def bootstrap_admin(app) -> None:
    """
    Create the admin account on first start from ADMIN_USERNAME and
    ADMIN_INITIAL_PASSWORD. After that the database copy is the only one
    that counts, and changing .env has no effect on an existing account.
    """
    with app.app_context():
        username = app.config["ADMIN_USERNAME"]
        if models.get_admin(username):
            return

        initial_password = app.config.get("ADMIN_INITIAL_PASSWORD") or ""
        if not initial_password:
            app.logger.warning("No admin account exists and ADMIN_INITIAL_PASSWORD is not set; admin login disabled.")
            return

        # The development password is fine on a laptop, but refuse to start
        # a production deployment with it.
        if app.config.get("APP_ENV") == "production" and password_problems(initial_password, username):
            raise RuntimeError("ADMIN_INITIAL_PASSWORD is too weak for production. Set a strong password.")

        models.create_admin(username, hash_password(initial_password))
        app.logger.info("Administrator account '%s' created from environment configuration.", username)


def participant_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        participant_id = session.get("participant_id")
        # Also check the database, in case the participant was deleted
        # (withdrawal) while their browser session was still open.
        if not participant_id or not models.get_participant(participant_id):
            session.pop("participant_id", None)
            flash("Please read the participant information and provide consent before using PhisScan.", "info")
            return redirect(url_for("participant.consent"))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        admin_id = session.get("admin_id")
        idle_limit = current_app.config["ADMIN_SESSION_MINUTES"] * 60
        timed_out = time.time() - session.get("admin_last_seen", 0) > idle_limit

        if not admin_id or timed_out:
            for key in ("admin_id", "admin_username", "admin_last_seen"):
                session.pop(key, None)
            if admin_id:
                flash("Your administrator session has expired. Please log in again.", "info")
            return redirect(url_for("admin.login"))

        session["admin_last_seen"] = time.time()
        return view(*args, **kwargs)
    return wrapper
