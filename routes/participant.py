"""
The participant journey around the scanner: welcome page, information and
consent, the awareness page, the debrief, and finishing the session.

The order (welcome -> consent -> dashboard -> debrief -> finish) follows the
study protocol. Every page after consent is protected by
participant_required, so the scanner can't be reached by typing a URL
directly.
"""

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from database import models
from security.auth import participant_required

bp = Blueprint("participant", __name__)

# The wording must match the approved consent form exactly, so it lives
# here in one place. The keys become the checkbox names (consent_<key>).
CONSENT_STATEMENTS = [
    ("age", "I confirm that I am 18 years of age or older."),
    ("read", "I have read and understood the participant information provided."),
    ("purpose", "I understand the purpose of this research and what my participation involves."),
    ("voluntary", "I understand that my participation is voluntary."),
    ("stop", "I understand that I may stop participating without providing a reason."),
    ("simulated", "I understand that this is a simulated research prototype and no real phishing websites will be accessed."),
    ("no_sensitive", "I understand that I should not submit personal, financial, or sensitive information."),
    ("anonymous_use", "I understand that anonymous usage statistics from my session may be used for academic research."),
    ("protocol", "I understand that the research data will be stored and handled according to the approved research protocol."),
    ("agree", "I agree to participate in this research."),
]


def _mark_visitor(status: str) -> None:
    if session.get("visitor_session_id"):
        models.touch_visitor_session(session["visitor_session_id"], status)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.route("/consent", methods=["GET", "POST"])
def consent():
    # Someone who has already consented doesn't need to see the form again.
    if session.get("participant_id") and models.get_participant(session["participant_id"]):
        return redirect(url_for("detection.dashboard"))

    errors = []
    if request.method == "POST":
        # The button is disabled in the browser until every box is ticked,
        # but that's only JavaScript. This is the check that actually counts.
        missing = [key for key, _ in CONSENT_STATEMENTS if request.form.get(f"consent_{key}") != "yes"]
        if missing:
            errors.append("Please confirm every statement to take part. "
                          "If you do not wish to take part, choose Decline.")
        else:
            session["participant_id"] = models.create_participant()
            session.pop("finished", None)
            _mark_visitor("consented")
            return redirect(url_for("detection.dashboard"))

    ticked = {key for key, _ in CONSENT_STATEMENTS if request.form.get(f"consent_{key}") == "yes"}
    page = render_template(
        "consent.html",
        statements=CONSENT_STATEMENTS,
        errors=errors,
        checked=ticked,   # keep the ticks if the form comes back with an error
        withdrawal_info=current_app.config["WITHDRAWAL_INFO"],
        researcher_contact=current_app.config["RESEARCHER_CONTACT"],
        supervisor_contact=current_app.config["SUPERVISOR_CONTACT"],
    )
    return page, (400 if errors else 200)


@bp.post("/decline")
def decline():
    # No reason is asked for and nothing about the person is stored. The
    # visitor session is just marked as declined so the admin can see how
    # many people chose not to take part.
    _mark_visitor("declined")
    session.pop("participant_id", None)
    return redirect(url_for("participant.declined"))


@bp.get("/declined")
def declined():
    return render_template(
        "message.html",
        title="Participation declined",
        heading="Thank you for your time",
        body=["You have chosen not to take part. No research data has been collected and you "
              "do not need to give a reason.",
              "You can close this window."],
    )


@bp.get("/awareness")
@participant_required
def awareness():
    return render_template("awareness.html")


@bp.get("/debrief")
@participant_required
def debrief():
    return render_template(
        "debrief.html",
        withdrawal_info=current_app.config["WITHDRAWAL_INFO"],
        researcher_contact=current_app.config["RESEARCHER_CONTACT"],
    )


@bp.post("/finish")
@participant_required
def finish():
    participant_id = session["participant_id"]
    models.complete_participant(participant_id)
    _mark_visitor("completed")

    # End the research session but keep the stored records. The ID is kept
    # just long enough to show it on the thank-you page, so the participant
    # can note it down for a withdrawal request.
    session.pop("participant_id", None)
    session["finished"] = participant_id
    return redirect(url_for("participant.finished"))


@bp.get("/finished")
def finished():
    participant_id = session.get("finished")
    if not participant_id:
        return redirect(url_for("participant.index"))
    return render_template("finished.html", finished_id=participant_id)
