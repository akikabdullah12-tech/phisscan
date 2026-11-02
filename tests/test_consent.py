"""Consent gating, declining, finishing, and how visitor sessions are counted."""

from database import models
from routes.participant import CONSENT_STATEMENTS


def test_dashboard_requires_consent(client):
    for path in ("/dashboard", "/debrief", "/awareness"):
        r = client.get(path)
        assert r.status_code == 302 and "/consent" in r.headers["Location"]
    assert client.post("/scan/url", data={"url": "https://example.com"}).status_code == 302


def test_partial_consent_rejected(client, app):
    data = {f"consent_{k}": "yes" for k, _ in CONSENT_STATEMENTS[:-1]}
    r = client.post("/consent", data=data)
    assert r.status_code == 400
    with app.app_context():
        assert models.admin_stats()["consents"] == 0
    assert client.get("/dashboard").status_code == 302


def test_full_consent_creates_anonymous_participant(consented, app):
    r = consented.get("/dashboard")
    assert r.status_code == 200
    with consented.session_transaction() as s:
        pid = s["participant_id"]
    assert pid == "PASV1001"
    with app.app_context():
        p = models.get_participant(pid)
        assert p["consent_status"] == "given" and p["research_status"] == "active"


def test_decline_ends_session(client, app):
    client.get("/consent")
    r = client.post("/decline")
    assert r.status_code == 302
    assert client.get("/dashboard").status_code == 302
    with app.app_context():
        assert models.admin_stats()["declined_sessions"] == 1


def test_visitor_counting_ignores_refresh(client, app):
    for _ in range(5):
        client.get("/")
    with app.app_context():
        assert models.public_stats()["visitors"] == 1
    other = app.test_client()
    other.get("/")
    with app.app_context():
        assert models.public_stats()["visitors"] == 2


def test_finish_completes_and_clears_session(consented, app):
    with consented.session_transaction() as s:
        pid = s["participant_id"]
    r = consented.post("/finish")
    assert r.status_code == 302
    assert consented.get("/dashboard").status_code == 302
    with app.app_context():
        assert models.get_participant(pid)["research_status"] == "completed"


def test_participant_ids_are_sequential_and_never_reused(app):
    from routes.participant import CONSENT_STATEMENTS
    consent = {f"consent_{k}": "yes" for k, _ in CONSENT_STATEMENTS}
    ids = []
    for _ in range(3):
        c = app.test_client()
        c.post("/consent", data=consent)
        with c.session_transaction() as s:
            ids.append(s["participant_id"])
    assert ids == ["PASV1001", "PASV1002", "PASV1003"]

    # Withdrawing PASV1003 must not free its number for the next person.
    with app.app_context():
        models.delete_participant("PASV1003")
    c = app.test_client()
    c.post("/consent", data=consent)
    with c.session_transaction() as s:
        assert s["participant_id"] == "PASV1004"
