"""
CSRF protection for every form in the app.

Each browser session gets a random token. Every POST has to send it back
(as the hidden csrf_token field, or an X-CSRF-Token header), and anything
without a matching token is rejected before it reaches a view. This stops
another website from submitting forms on a logged-in admin's behalf, e.g. a
hidden form that deletes a participant.

It's small enough that writing it here was simpler than adding Flask-WTF
just for this.
"""

import hmac
import secrets

from flask import abort, current_app, request, session

SESSION_KEY = "_csrf_token"
UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def generate_csrf_token() -> str:
    """Called from the templates. Creates the token the first time it's needed."""
    token = session.get(SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[SESSION_KEY] = token
    return token


def csrf_protect() -> None:
    """before_request hook."""
    if request.method not in UNSAFE_METHODS or not current_app.config.get("CSRF_ENABLED", True):
        return

    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    expected = session.get(SESSION_KEY, "")

    # compare_digest takes the same time however many characters match,
    # so the token can't be guessed one character at a time.
    if not expected or not submitted or not hmac.compare_digest(str(submitted), str(expected)):
        abort(400, description="Your session has expired or the form was invalid. "
                               "Please go back, refresh the page and try again.")
