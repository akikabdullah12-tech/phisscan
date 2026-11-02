"""Admin access control, login hardening, CSRF, headers and the admin pages."""

from app import create_app
from database import models
from tests.conftest import ADMIN_PASSWORD, get_csrf


def _login(client):
    return client.post("/admin/login", data={"username": "akik", "password": ADMIN_PASSWORD})


def test_admin_routes_protected(client):
    for path in ("/admin/", "/admin/models", "/admin/evaluation/url_roc_curves.png"):
        response = client.get(path)
        assert response.status_code == 302 and "/admin/login" in response.headers["Location"]


def test_removed_pages_are_gone(client):
    _login(client)
    for path in ("/admin/questionnaire", "/admin/export/questionnaire.csv", "/admin/password", "/questionnaire"):
        assert client.get(path).status_code == 404, path


def test_login_success_and_logout(client):
    assert _login(client).status_code == 302
    assert client.get("/admin/").status_code == 200
    client.post("/admin/logout")
    assert client.get("/admin/").status_code == 302


def test_wrong_password_and_rate_limit(client):
    for _ in range(5):
        assert client.post("/admin/login", data={"username": "akik", "password": "wrong"}).status_code == 401
    # Locked out now, even with the right password.
    assert _login(client).status_code == 429


def test_password_stored_hashed(app):
    with app.app_context():
        stored = models.get_admin("akik")["password_hash"]
    assert ADMIN_PASSWORD not in stored
    assert stored.startswith(("scrypt:", "pbkdf2:"))


def test_csrf_enforced(tmp_path):
    app = create_app({"TESTING": True, "DATABASE_PATH": str(tmp_path / "c.sqlite3"), "SECRET_KEY": "x",
                      "ADMIN_INITIAL_PASSWORD": ADMIN_PASSWORD, "CSRF_ENABLED": True})
    client = app.test_client()
    assert client.post("/admin/login", data={"username": "akik", "password": ADMIN_PASSWORD}).status_code == 400

    token = get_csrf(client, "/admin/login")
    assert token
    response = client.post("/admin/login",
                           data={"username": "akik", "password": ADMIN_PASSWORD, "csrf_token": token})
    assert response.status_code == 302


def test_security_headers(client):
    response = client.get("/")
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_session_cookie_flags(client):
    cookie = client.get("/").headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_database_not_served(client):
    assert client.get("/instance/phisscan.sqlite3").status_code == 404
    assert client.get("/static/../instance/phisscan.sqlite3").status_code == 404


def test_no_admin_link_for_participants(consented):
    assert b"/admin" not in consented.get("/dashboard").data


def test_dashboard_lists_participants_and_withdrawal_deletes_them(client, consented, app):
    with consented.session_transaction() as s:
        participant_id = s["participant_id"]

    _login(client)
    page = client.get("/admin/")
    assert page.status_code == 200 and participant_id.encode() in page.data

    assert client.post(f"/admin/participant/{participant_id}/delete").status_code == 302
    with app.app_context():
        assert models.get_participant(participant_id) is None


def test_admin_pages_render(client, consented):
    _login(client)
    for path in ("/admin/", "/admin/models"):
        assert client.get(path).status_code == 200, path
