"""
PhisScan - Machine Learning-Based Phishing Detection and User Awareness System.
MSc research prototype, Abertay University (CMP504).

Start it with:

    python app.py            then open http://127.0.0.1:5000

create_app() builds the Flask app. Using a factory rather than a global app
lets the tests create a fresh app with its own temporary database each time.
"""

from __future__ import annotations

import logging
import os
import time

from flask import Flask, render_template, request, session
from werkzeug.exceptions import HTTPException

from config import Config
from database import database, models
from security.auth import bootstrap_admin
from security.csrf import csrf_protect, generate_csrf_token
from security.rate_limit import LoginRateLimiter

# Fairly strict: no inline scripts or styles, nothing loaded from other
# origins, and the pages can't be framed (clickjacking). The templates and
# main.js were written to work within these rules.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("SECRET_KEY") and not app.config.get("TESTING"):
        app.logger.warning("SECRET_KEY not set - using a random key, so sessions reset on restart. See .env.example.")

    database.init_app(app)
    bootstrap_admin(app)
    app.extensions["login_limiter"] = LoginRateLimiter(
        app.config["LOGIN_MAX_ATTEMPTS"], app.config["LOGIN_WINDOW_MINUTES"] * 60)

    # Imported here rather than at the top so the blueprints load after the
    # app is configured.
    from routes.admin import bp as admin_bp
    from routes.detection import bp as detection_bp
    from routes.participant import bp as participant_bp

    for blueprint in (participant_bp, detection_bp, admin_bp):
        app.register_blueprint(blueprint)

    _register_request_hooks(app)
    _register_template_helpers(app)
    _register_error_pages(app)

    @app.cli.command("purge-expired")
    def purge_expired():
        """Apply DATA_RETENTION_DAYS from the command line (handy for a scheduled job)."""
        days = app.config["DATA_RETENTION_DAYS"]
        removed = models.purge_older_than(days)
        print(f"Removed {removed} participant record(s) older than {days} days.")

    return app


def _register_request_hooks(app: Flask) -> None:
    @app.before_request
    def before_each_request():
        csrf_protect()
        if request.endpoint in (None, "static"):
            return
        session.permanent = True   # use PERMANENT_SESSION_LIFETIME instead of "until browser closes"
        _track_visitor_session()

    @app.after_request
    def add_security_headers(response):
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Reports and admin pages contain things we don't want sitting in a
        # shared computer's browser cache. Static files can be cached normally.
        if request.endpoint != "static":
            response.headers["Cache-Control"] = "no-store"
        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def _register_template_helpers(app: Flask) -> None:
    @app.context_processor
    def template_globals():
        return {
            "csrf_token": generate_csrf_token,
            "researcher": app.config["RESEARCHER_NAME"],
            "university": app.config["UNIVERSITY"],
            "is_participant": bool(session.get("participant_id")),
            "participant_id": session.get("participant_id"),
            "is_admin": bool(session.get("admin_id")),
        }

    @app.template_filter("pct")
    def as_percentage(value, digits=0):
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "-"

    @app.template_filter("prob")
    def as_probability(value):
        # A displayed "100%" reads as a guarantee, which no model can give,
        # so the extremes are softened.
        try:
            p = float(value)
        except (TypeError, ValueError):
            return "-"
        if p >= 0.995:
            return "over 99%"
        if p < 0.005:
            return "under 1%"
        return f"{p * 100:.0f}%"

    @app.template_filter("wclass")
    def width_class(value):
        # The CSP blocks style="width: 42%", so bar widths are done with
        # predefined classes (w0, w5 ... w100) from style.css.
        try:
            fraction = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            fraction = 0.0
        return f"w{int(round(fraction * 20) * 5)}"


def _register_error_pages(app: Flask) -> None:
    # Friendly messages only: stack traces and internal details stay in the
    # server log, never in the browser.
    friendly = {
        403: "You do not have permission to view this page.",
        404: "The page you requested could not be found.",
        405: "This action is not allowed.",
        413: "The submitted content is too large.",
        429: "Too many requests. Please wait and try again.",
    }

    @app.errorhandler(HTTPException)
    def http_error(err):
        if err.code == 400:
            # The CSRF check provides its own (safe) explanation; other 400s don't.
            message = err.description if "session" in str(err.description) else "The request could not be processed."
        else:
            message = friendly.get(err.code, "Something went wrong.")
        return render_template("error.html", code=err.code, message=message), err.code

    @app.errorhandler(Exception)
    def unexpected_error(err):
        app.logger.exception("Unhandled error")
        return render_template("error.html", code=500,
                               message="An unexpected error occurred. Please try again."), 500


def _track_visitor_session() -> None:
    """
    Count anonymous visitor *sessions* for the dashboard.

    A random session ID is created on the first request and stored in the
    session cookie. Later requests only bump its last-activity time (at most
    once a minute, to avoid a database write on every click). Refreshing
    doesn't add visitors. This counts browser sessions, not people, and the
    dashboard says so.
    """
    session_id = session.get("visitor_session_id")
    now = time.time()

    if session_id:
        if now - session.get("visitor_touched", 0) < 60:
            return
        if models.touch_visitor_session(session_id):
            session["visitor_touched"] = now
            return
        # The row is gone (e.g. the database was reset), so start a new one.

    session["visitor_session_id"], _visitor_id = models.create_visitor_session()
    session["visitor_touched"] = now


if __name__ == "__main__":
    application = create_app()
    application.run(
        host=os.environ.get("HOST", "127.0.0.1"),   # local only unless deliberately changed
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
