"""
The researcher's admin area, under /admin.

Two pages: an overview (study statistics and participant activity) and the
model evaluation results. The area isn't linked from any participant page,
every view apart from the login form needs an authenticated admin session,
and state-changing actions are POST requests protected by the CSRF token.
"""

from __future__ import annotations

import re
import time

from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   send_from_directory, session, url_for)

from database import models
from ml import evaluate_models as ev
from ml.email_predictor import predictor as email_predictor
from ml.url_predictor import predictor as url_predictor
from security.auth import admin_required, verify_password

bp = Blueprint("admin", __name__, url_prefix="/admin")

# Participant IDs look like PASV1001. Anything else in the filter box or a
# URL is rejected before it gets near a query.
_PARTICIPANT_ID_RE = re.compile(r"^[A-Za-z0-9-]{0,20}$")


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------

@bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))
    if request.method == "GET":
        return render_template("admin_login.html", error=None)

    username = request.form.get("username", "").strip()[:64]
    password = request.form.get("password", "")[:256]

    limiter = current_app.extensions["login_limiter"]
    limiter_key = f"{request.remote_addr}|{username.lower()}"
    blocked, wait_seconds = limiter.is_blocked(limiter_key)
    if blocked:
        minutes = max(1, wait_seconds // 60)
        return render_template("admin_login.html",
                               error=f"Too many failed attempts. Please try again in {minutes} minute(s)."), 429

    admin = models.get_admin(username) if username else None
    if not verify_password(admin["password_hash"] if admin else None, password):
        limiter.register_failure(limiter_key)
        # Same message for "no such user" and "wrong password", so it can't
        # be used to find out which usernames exist.
        return render_template("admin_login.html", error="Invalid username or password."), 401

    limiter.reset(limiter_key)

    # Start a clean session on login so a session ID planted before login
    # can't be reused afterwards (session fixation). The anonymous visitor
    # ID is carried over so this browser isn't counted as a new visitor.
    visitor = session.get("visitor_session_id"), session.get("visitor_touched")
    session.clear()
    session["visitor_session_id"], session["visitor_touched"] = visitor
    session["admin_id"] = admin["admin_id"]
    session["admin_username"] = admin["username"]
    session["admin_last_seen"] = time.time()
    models.record_admin_login(admin["admin_id"])
    return redirect(url_for("admin.dashboard"))


@bp.post("/logout")
def logout():
    for key in ("admin_id", "admin_username", "admin_last_seen"):
        session.pop(key, None)
    flash("You have been logged out.", "info")
    return redirect(url_for("admin.login"))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

@bp.get("/")
@admin_required
def dashboard():
    id_filter = request.args.get("participant", "").strip()
    if not _PARTICIPANT_ID_RE.match(id_filter):
        id_filter = ""
    return render_template(
        "admin_dashboard.html",
        page="overview",
        stats=models.admin_stats(),
        participants=models.participant_activity(id_filter or None),
        filter_value=id_filter,
        retention_days=current_app.config["DATA_RETENTION_DAYS"],
    )


@bp.post("/participant/<participant_id>/delete")
@admin_required
def delete_participant(participant_id):
    """Handle a withdrawal request: remove everything stored for this ID."""
    if not _PARTICIPANT_ID_RE.match(participant_id):
        abort(404)
    if models.delete_participant(participant_id):
        flash(f"All records for participant {participant_id} were deleted (withdrawal).", "success")
    else:
        flash("Participant not found.", "warning")
    return redirect(url_for("admin.dashboard"))


@bp.post("/retention")
@admin_required
def apply_retention():
    days = current_app.config["DATA_RETENTION_DAYS"]
    if days <= 0:
        flash("No retention period is configured (DATA_RETENTION_DAYS = 0).", "info")
    else:
        removed = models.purge_older_than(days)
        flash(f"Retention policy applied: {removed} participant record(s) older than {days} days removed.", "success")
    return redirect(url_for("admin.dashboard"))


# ---------------------------------------------------------------------------
# Model evaluation results
# ---------------------------------------------------------------------------

@bp.get("/models")
@admin_required
def model_results():
    # Everything on this page comes from the JSON files written by the
    # training scripts. Nothing is recalculated or typed in by hand.
    return render_template(
        "admin_models.html",
        page="models",
        url_results=ev.load_results("url"),
        url_full=ev.load_results("url_full"),
        email_results=ev.load_results("email"),
        url_info=url_predictor.info(),
        email_info=email_predictor.info(),
    )


@bp.get("/evaluation/<path:filename>")
@admin_required
def evaluation_file(filename):
    # Only the chart images, and only plain names like url_roc_curves.png.
    # This rules out "../" tricks for reading other files.
    if not re.fullmatch(r"[a-z_]+\.png", filename):
        abort(404)
    return send_from_directory(ev.EVAL_DIR, filename)
