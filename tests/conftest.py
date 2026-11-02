"""
Shared pytest fixtures.

Each test gets a brand-new app with its own temporary SQLite file, so tests
can't affect each other or the real database in instance/. CSRF is switched
off for most tests to keep them readable; test_admin_security.py has a
separate test that runs with it switched on.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from ml.email_predictor import predictor as email_predictor  # noqa: E402
from ml.url_predictor import predictor as url_predictor  # noqa: E402

ADMIN_PASSWORD = "test-Password-123!"


@pytest.fixture
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(tmp_path / "test.sqlite3"),
        "SECRET_KEY": "test-secret",
        "ADMIN_USERNAME": "akik",
        "ADMIN_INITIAL_PASSWORD": ADMIN_PASSWORD,
        "CSRF_ENABLED": False,
    })
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def consented(client):
    """A client that has already ticked every consent box."""
    from routes.participant import CONSENT_STATEMENTS
    data = {f"consent_{k}": "yes" for k, _ in CONSENT_STATEMENTS}
    r = client.post("/consent", data=data)
    assert r.status_code == 302
    return client


requires_url_model = pytest.mark.skipif(not url_predictor.is_ready(), reason="URL model not trained")
requires_email_model = pytest.mark.skipif(not email_predictor.is_ready(), reason="Email model not trained")


def get_csrf(client, path):
    html = client.get(path).get_data(as_text=True)
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return m.group(1) if m else None
