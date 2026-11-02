"""
The scanner itself: the dashboard, the two analysis forms and the report pages.

Privacy design: a submitted URL or email is never written to the database.
Only an anonymous record that a scan happened goes in (scan ID, participant
ID, time, prediction, model version). The full report needed to draw the
results page is kept briefly in server memory, tied to the participant who
made it, and thrown away after REPORT_CACHE_MINUTES.

Forms use Post/Redirect/Get: the POST runs the analysis and redirects to
the report URL. Refreshing the report therefore doesn't resubmit the scan or
count it twice.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from database import models
from ml.email_predictor import EmailValidationError, predictor as email_predictor
from ml.url_features import URLValidationError
from ml.url_predictor import ModelNotReadyError, predictor as url_predictor
from security.auth import participant_required

bp = Blueprint("detection", __name__)


class ReportCache:
    """
    A small in-memory store for scan reports. It's never persisted, so a
    server restart simply clears it, which is acceptable for reports the
    participant looks at once.
    """

    def __init__(self, max_items: int = 2000):
        self._items: dict[str, tuple[str, float, dict]] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def put(self, scan_id: str, owner: str, report: dict) -> None:
        with self._lock:
            if len(self._items) >= self.max_items:
                # Drop the oldest so memory can't grow without limit.
                oldest = min(self._items, key=lambda k: self._items[k][1])
                del self._items[oldest]
            self._items[scan_id] = (owner, time.time(), report)

    def get(self, scan_id: str, owner: str, ttl_seconds: int) -> dict | None:
        with self._lock:
            item = self._items.get(scan_id)
            if item is None:
                return None
            item_owner, created, report = item
            if time.time() - created > ttl_seconds:
                del self._items[scan_id]
                return None
            # Scan IDs are random, but even with a valid ID one participant
            # can never open another participant's report.
            return report if item_owner == owner else None


reports = ReportCache()


def _report_ttl() -> int:
    return current_app.config["REPORT_CACHE_MINUTES"] * 60


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _render_dashboard(status: int = 200, **context):
    page = render_template(
        "dashboard.html",
        stats=models.public_stats(),
        url_ready=url_predictor.is_ready(),
        email_ready=email_predictor.is_ready(),
        url_info=url_predictor.info(),
        email_info=email_predictor.info(),
        **context,
    )
    return page, status


def _report_expired():
    page = render_template(
        "message.html",
        title="Report unavailable",
        heading="Report no longer available",
        body=["Scan reports are kept only temporarily and are not stored. Please run the analysis again."],
        back_to_dashboard=True,
    )
    return page, 404


@bp.get("/dashboard")
@participant_required
def dashboard():
    return _render_dashboard(active=request.args.get("tab", "home"))


# ---------------------------------------------------------------------------
# URL analysis
# ---------------------------------------------------------------------------

@bp.post("/scan/url")
@participant_required
def scan_url():
    submitted = request.form.get("url", "")
    try:
        result = url_predictor.predict(submitted)
    except (URLValidationError, ModelNotReadyError) as exc:
        # Show the form again with the message and the user's input, so
        # they can fix a typo instead of retyping everything.
        status = 400 if isinstance(exc, URLValidationError) else 503
        return _render_dashboard(status, active="url", url_error=str(exc), url_value=submitted[:2048])

    participant_id = session["participant_id"]
    scan_id = models.new_id("URL", 5)
    # Only counted once the prediction has succeeded, so failed attempts
    # don't inflate the statistics.
    models.record_scan("url", scan_id, participant_id, result["prediction"], result["model_version"])

    result.update(scan_id=scan_id, scan_type="URL", timestamp=_timestamp())
    reports.put(scan_id, participant_id, result)
    return redirect(url_for("detection.url_report", scan_id=scan_id), code=303)


@bp.get("/report/url/<scan_id>")
@participant_required
def url_report(scan_id):
    report = reports.get(scan_id, session["participant_id"], _report_ttl())
    if report is None:
        return _report_expired()
    return render_template("url_report.html", r=report)


# ---------------------------------------------------------------------------
# Email analysis
# ---------------------------------------------------------------------------

@bp.post("/scan/email")
@participant_required
def scan_email():
    subject = request.form.get("subject", "")
    sender = request.form.get("sender", "")
    body = request.form.get("body", "")
    try:
        result = email_predictor.predict(subject, sender, body)
    except (EmailValidationError, ModelNotReadyError) as exc:
        status = 400 if isinstance(exc, EmailValidationError) else 503
        previous = {"subject": subject[:300], "sender": sender[:254], "body": body[:50_000]}
        return _render_dashboard(status, active="email", email_error=str(exc), email_values=previous)

    participant_id = session["participant_id"]
    scan_id = models.new_id("EML", 5)
    models.record_scan("email", scan_id, participant_id, result["prediction"], result["model_version"])

    result.update(scan_id=scan_id, scan_type="Email", timestamp=_timestamp())
    reports.put(scan_id, participant_id, result)
    return redirect(url_for("detection.email_report", scan_id=scan_id), code=303)


@bp.get("/report/email/<scan_id>")
@participant_required
def email_report(scan_id):
    report = reports.get(scan_id, session["participant_id"], _report_ttl())
    if report is None:
        return _report_expired()
    flagged = [u for u in report["urls"] if u.get("is_phishing")]
    return render_template("email_report.html", r=report, flagged_urls=flagged)
