"""Scan counters, report access and the privacy of submitted URLs."""

import sqlite3

from database import models
from tests.conftest import requires_url_model


@requires_url_model
def test_scan_counters_and_report_privacy(consented, app):
    # A rejected URL must not be counted as a scan.
    consented.post("/scan/url", data={"url": "not a url"})
    with app.app_context():
        assert models.public_stats()["url_scans"] == 0

    response = consented.post("/scan/url", data={"url": "https://example.com"})
    assert response.status_code == 303
    with app.app_context():
        stats = models.public_stats()
        assert stats["url_scans"] == 1 and stats["total_scans"] == 1

    # The submitted URL should not appear anywhere in the database.
    dump = "\n".join(sqlite3.connect(app.config["DATABASE_PATH"]).iterdump())
    assert "example.com" not in dump

    report_url = response.headers["Location"]
    report = consented.get(report_url)
    assert report.status_code == 200 and b"URL Analysis Report" in report.data
    assert b"Finish Session" in report.data

    # Someone else who gets hold of the report link is sent to the consent page.
    assert app.test_client().get(report_url).status_code == 302


def test_awareness_page_links_back_to_homepage(consented):
    page = consented.get("/awareness").get_data(as_text=True)
    assert "Back to Homepage" in page
    assert "Back to Dashboard" not in page


def test_no_questionnaire_in_participant_pages(consented):
    for path in ("/dashboard", "/awareness", "/debrief"):
        assert b"uestionnaire" not in consented.get(path).data, path
