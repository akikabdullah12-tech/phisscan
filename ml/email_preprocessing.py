"""
Email input handling: validation, HTML stripping, text cleaning for TF-IDF,
link extraction and the rule-based email warnings.

Pasted emails are treated as untrusted text from start to finish. HTML is
stripped with regular expressions rather than parsed into a DOM, nothing is
ever rendered, scripts are discarded, and links are only collected as strings
so the URL model can look at them. Attachments simply aren't supported.

clean_text() is shared with the training script on purpose. If the two ever
cleaned text differently, the model would be scoring something it never saw
during training.
"""

from __future__ import annotations

import html
import re

from ml.lists import FREE_MAIL_PROVIDERS, URL_SHORTENERS
from ml.url_features import URLValidationError, impersonated_brands, parse_url

# Limits are generous for a pasted email but stop someone posting a novel.
MAX_EMAIL_CHARS = 50_000
MAX_SUBJECT_CHARS = 300
MAX_SENDER_CHARS = 254          # RFC 5321 maximum address length
MAX_URLS_ANALYSED = 10          # each link costs a model call, so cap it


class EmailValidationError(ValueError):
    """The email can't be analysed. The message is safe to show to the user."""


_SCRIPT_STYLE_RE = re.compile(r"<(script|style|head)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_END_RE = re.compile(r"<\s*(br|/p|/div|/li|/tr|/h\d)\b[^>]*>", re.I)
_URL_RE = re.compile(r"""(?:https?://|www\.)[^\s<>"'()\[\]{}]+""", re.I)
_HREF_RE = re.compile(r"""href\s*=\s*["']?([^"'\s>]+)""", re.I)
_ANCHOR_RE = re.compile(r"""<a\b[^>]*href\s*=\s*["']?([^"'\s>]+)[^>]*>(.*?)</a\s*>""", re.I | re.S)
_ADDRESS_RE = re.compile(r"\S+@\S+")
_SENDER_RE = re.compile(r"^[A-Za-z0-9._%+'-]{1,64}@([A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,63}$")


# ---------------------------------------------------------------------------
# Turning an email into model input
# ---------------------------------------------------------------------------

def html_to_text(raw: str) -> str:
    """Reduce (possibly HTML) email content to readable plain text."""
    # Script/style blocks go first, otherwise their contents would survive as
    # "text" once the tags around them are removed.
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _BLOCK_END_RE.sub("\n", text)     # keep paragraph breaks readable
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Normalise text before TF-IDF.

    Links and addresses are replaced with placeholder words instead of being
    deleted. Their exact characters are noise, but the fact that an email
    contains a link at all is useful information for the model.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _ADDRESS_RE.sub(" emailtoken ", text)
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def model_input(subject: str, body_text: str) -> str:
    # Many emails in the training data start with "subject: ...", so putting
    # the subject in front of the body keeps live input close to that shape.
    return clean_text(f"{subject or ''} {body_text or ''}")


def validate_email_input(subject: str | None, sender: str | None, body: str | None) -> dict:
    """Check the three form fields and return them in a tidy dict."""
    subject = (subject or "").strip()
    sender = (sender or "").strip()
    body = (body or "").strip()

    if not body:
        raise EmailValidationError("Please paste the email message to analyse.")
    if len(body) > MAX_EMAIL_CHARS:
        raise EmailValidationError(f"The email message is too long (maximum {MAX_EMAIL_CHARS:,} characters).")
    if len(subject) > MAX_SUBJECT_CHARS:
        raise EmailValidationError(f"The subject is too long (maximum {MAX_SUBJECT_CHARS} characters).")

    address, display_name = "", ""
    if sender:
        # People usually copy the sender as it appears in their mail client,
        # e.g. 'PayPal Security <service@paypa1.com>', so accept that form too.
        match = re.search(r"<([^<>]+)>\s*$", sender)
        address = (match.group(1) if match else sender).strip()
        if len(address) > MAX_SENDER_CHARS or not _SENDER_RE.match(address):
            raise EmailValidationError("The sender email address does not look valid. Leave it blank if unknown.")
        if match:
            display_name = sender[: match.start()].strip().strip('"')

    plain = html_to_text(body)
    if len(clean_text(plain)) < 3:
        raise EmailValidationError("The email message does not contain enough text to analyse.")

    return {
        "subject": subject,
        "sender": address,
        "sender_display": display_name,
        "raw_body": body,
        "body_text": plain,
    }


def extract_urls(raw_body: str) -> list[str]:
    """
    Collect the links in an email, from both href attributes and plain text.
    The hrefs matter most: that's where the real destination of an HTML link
    is, and it's often different from the text the reader sees.
    """
    visible_text = html.unescape(_TAG_RE.sub(" ", raw_body))
    candidates = _HREF_RE.findall(raw_body) + _URL_RE.findall(visible_text)

    urls: list[str] = []
    for candidate in candidates:
        url = html.unescape(candidate).strip().rstrip(".,;:!?)'\"")
        if url.lower().startswith(("mailto:", "tel:", "#", "javascript:", "data:")):
            continue
        if url not in urls:
            urls.append(url)
    return urls[:MAX_URLS_ANALYSED]


# ---------------------------------------------------------------------------
# Rule-based warnings
#
# These mirror the standard "how to spot a phishing email" advice. They are
# shown as prompts for caution, never as proof, because genuine emails use
# many of the same phrases ("your account", "verify your email", ...).
# ---------------------------------------------------------------------------

PHRASE_GROUPS = {
    "urgent": ("Urgent or time-pressure language", [
        "urgent", "immediately", "within 24 hours", "within 48 hours", "act now", "as soon as possible",
        "final notice", "expires today", "will expire", "last chance", "limited time", "asap", "right away",
        "action required", "respond now", "deadline"]),
    "sensitive": ("Request for sensitive information", [
        "password", "pin number", "your pin", "social security", "bank account", "credit card",
        "card number", "cvv", "security code", "login details", "credentials", "date of birth",
        "sort code", "security question", "one-time code", "otp", "passcode"]),
    "verify": ("Account verification request", [
        "verify your account", "confirm your account", "verify your identity", "update your account",
        "validate your account", "confirm your details", "confirm your identity", "reactivate",
        "unlock your account", "update your payment", "verify your information", "update your information",
        "confirm your information", "re-confirm", "verify your email", "confirm your password"]),
    "threat": ("Threatening or alarming language", [
        "suspended", "will be closed", "terminated", "legal action", "locked", "unauthorized",
        "unauthorised", "unusual activity", "suspicious activity", "penalty", "deactivated",
        "restricted", "permanently disabled", "compromised", "breach"]),
    "reward": ("Unexpected reward or prize", [
        "you have won", "you've won", "winner", "prize", "lottery", "claim your", "free gift",
        "inheritance", "million dollars", "cash reward", "selected to receive", "congratulations"]),
    "generic": ("Generic greeting", [
        "dear customer", "dear user", "dear valued", "dear account holder", "dear sir/madam",
        "dear member", "dear client", "dear beneficiary", "dear friend"]),
}

# Requests for credentials or "verify your account" are the most direct
# signs of credential phishing, so they get the stronger highlight.
_HIGH_RISK_GROUPS = {"sensitive", "verify"}

# Words that suggest a sender is presenting themselves as an organisation.
_ORGANISATION_WORDS = ["bank", "paypal", "amazon", "apple", "microsoft", "netflix", "hmrc", "royal mail", "dhl",
                       "support", "security", "billing", "account", "service", "team", "admin", "helpdesk"]

_BRANDS_MENTIONED = ("paypal", "amazon", "apple", "microsoft", "netflix", "hmrc", "dhl", "royal mail")


def _phrases_found(text: str, phrases: list[str]) -> list[str]:
    lowered = text.lower()
    # Word boundaries stop "pin" matching "shopping" and "otp" matching "hotpot".
    return [p for p in phrases if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", lowered)]


def email_indicators(email: dict, urls: list[str]) -> list[dict]:
    """Warning signs that are actually present in this email."""
    text = f"{email['subject']}\n{email['body_text']}"
    found: list[dict] = []

    for key, (title, phrases) in PHRASE_GROUPS.items():
        hits = _phrases_found(text, phrases)
        if hits:
            found.append({
                "title": title,
                "matches": hits[:5],
                "level": "high" if key in _HIGH_RISK_GROUPS else "warning",
                "detail": "Found: " + ", ".join(f"'{h}'" for h in hits[:5]),
            })

    mismatch = _mismatched_link(email["raw_body"])
    if mismatch:
        shown, real = mismatch
        found.append({"title": "Link text does not match its destination", "level": "high",
                      "detail": f"The link displays '{shown}' but points to '{real}'."})

    found.extend(_link_warnings(urls))

    if email.get("sender"):
        found.extend(_sender_warnings(email, text))

    return found


def _mismatched_link(raw_body: str) -> tuple[str, str] | None:
    """
    Find an <a> tag whose visible text shows one domain while the href goes
    to another, e.g. text "www.paypal.com" pointing at evil.xyz. This is one
    of the most reliable signs there is, so only the first case is reported.
    """
    for href, label_html in _ANCHOR_RE.findall(raw_body):
        label = html.unescape(_TAG_RE.sub("", label_html)).strip()
        domain_in_label = re.search(r"\S+\.[a-z]{2,}\S*", label, re.I)
        if not domain_in_label:
            continue   # ordinary link text like "click here"
        try:
            shown = parse_url(domain_in_label.group(0))
            real = parse_url(html.unescape(href))
        except URLValidationError:
            continue
        if shown.registered_domain != real.registered_domain:
            return shown.registered_domain, real.registered_domain
    return None


def _link_warnings(urls: list[str]) -> list[dict]:
    ip_hosts, shorteners = [], set()
    for url in urls:
        try:
            p = parse_url(url)
        except URLValidationError:
            continue
        if p.is_ip:
            ip_hosts.append(p.host)
        if p.host_no_www in URL_SHORTENERS:
            shorteners.add(p.host_no_www)

    warnings = []
    if ip_hosts:
        warnings.append({"title": "Link to a raw IP address", "level": "high",
                         "detail": "Links point to " + ", ".join(ip_hosts[:3]) + " instead of a named domain."})
    if shorteners:
        warnings.append({"title": "Shortened link", "level": "warning",
                         "detail": "Shortened links (" + ", ".join(sorted(shorteners)) + ") hide the real destination."})
    if len(urls) >= 5:
        warnings.append({"title": "Many links", "level": "info",
                         "detail": f"The message contains {len(urls)} or more links."})
    return warnings


def _sender_warnings(email: dict, text: str) -> list[dict]:
    sender = email["sender"]
    local_part, domain = sender.rsplit("@", 1)
    domain = domain.lower()
    display = (email.get("sender_display") or "").lower()
    warnings = []

    try:
        parsed = parse_url(domain)
        brands = impersonated_brands(parsed)
    except URLValidationError:
        parsed, brands = None, []

    if brands:
        warnings.append({"title": "Sender domain imitates a brand", "level": "high",
                         "detail": f"The sender domain '{domain}' mentions '{brands[0]}' "
                                   "but is not that organisation's domain."})

    if domain in FREE_MAIL_PROVIDERS and any(w in f"{display} {local_part.lower()}" for w in _ORGANISATION_WORDS):
        warnings.append({"title": "Organisation name from a free email account", "level": "warning",
                         "detail": f"The sender presents as an organisation but uses a free {domain} address."})

    if parsed and (sum(c.isdigit() for c in parsed.registered_domain) >= 3
                   or parsed.registered_domain.count("-") >= 2):
        warnings.append({"title": "Unusual sender domain", "level": "warning",
                         "detail": f"The sender domain '{domain}' contains many digits or hyphens."})

    # The message talks about a well-known company, but didn't come from it.
    # Skipped when the brand check above already fired, to avoid saying the
    # same thing twice.
    mentioned = [b for b in _BRANDS_MENTIONED if b in text.lower() and b.replace(" ", "") not in domain]
    if mentioned and not brands:
        warnings.append({"title": "Possible impersonation", "level": "warning",
                         "detail": f"The message mentions '{mentioned[0]}' but the sender domain is '{domain}'."})

    return warnings
