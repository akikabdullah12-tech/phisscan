"""
URL parsing, feature extraction and rule-based URL warnings.

Everything here works on the URL *string*. Nothing is resolved, fetched or
opened, which is a hard requirement of the ethics approval: participants may
paste real phishing links, and the app must never touch them.

There are two feature sets:

    "domain"  - what the deployed model uses. Features of the registered
                domain (google.com for accounts.google.com) plus a handful of
                host-level red flags (IP address, punycode, shortener, free
                hosting, brand impersonation).

    "full"    - the domain features plus whole-URL features (length, path,
                HTTPS, special characters...). Only used for the comparison
                experiment, not deployed.

Why not use the full URL? While exploring PhiUSIIL I found that every
legitimate URL is a bare "https://www.<domain>" homepage: no path, no query
string, no plain http. Phishing URLs often have all of those. A model trained
on whole-URL features scores ~99.8% simply by learning "has a path means
phishing", and then flags perfectly normal links like github.com/login.
The same thing happens with subdomains, since the legitimate samples almost
never have one. Restricting the model to the registered domain removes those
shortcuts. The path/HTTPS/subdomain signals aren't thrown away though: they
are still reported to the user as rule-based indicators, clearly separated
from the model's prediction.

Training and the web app both call extract_features(), so the model always
sees features computed in exactly the same way.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit

from ml.lists import (
    COMMON_BRANDS,
    FREE_HOSTING_DOMAINS,
    MULTI_PART_SUFFIXES,
    OFFICIAL_BRAND_DOMAINS,
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_TLDS,
    URL_SHORTENERS,
)

# Browsers cope with longer URLs, but 2048 is the long-standing practical
# limit and anything bigger is almost certainly junk or an attempt to make
# the server do pointless work.
MAX_URL_LENGTH = 2048

# The order of these lists matters: the saved model expects its columns in
# exactly this order, so only ever append to them (and retrain).
DOMAIN_FEATURES = [
    "domain_length",
    "domain_label_length",
    "num_hyphens_domain",
    "num_digits_domain",
    "digit_ratio_domain",
    "domain_entropy",
    "max_consonant_run",
    "vowel_ratio_domain",
    "tld_length",
    "suspicious_tld",
    "suspicious_keywords_domain",
    "has_ip",
    "has_punycode",
    "is_shortener",
    "free_hosting",
    "brand_in_host_mismatch",
]

FULL_ONLY_FEATURES = [
    "host_length",
    "num_dots",
    "num_subdomains",
    "url_length",
    "path_length",
    "query_length",
    "num_slashes",
    "num_special_chars",
    "num_at",
    "num_equals",
    "num_qmark",
    "num_ampersand",
    "num_percent_encoded",
    "num_digits_url",
    "num_hyphens_url",
    "url_entropy",
    "is_https",
    "has_www",
    "has_port",
    "double_slash_in_path",
    "suspicious_keywords_url",
]

FEATURE_SETS = {
    "domain": DOMAIN_FEATURES,
    "full": DOMAIN_FEATURES + FULL_ONLY_FEATURES,
}

# Plain-English names shown in the "what influenced the model" part of the
# report. Participants shouldn't have to decode "digit_ratio_domain".
FEATURE_DESCRIPTIONS = {
    "domain_length": "Length of the registered domain",
    "domain_label_length": "Length of the domain name (without the ending)",
    "num_hyphens_domain": "Hyphens in the domain name",
    "num_digits_domain": "Digits in the domain name",
    "digit_ratio_domain": "Proportion of digits in the domain name",
    "domain_entropy": "Randomness (Shannon entropy) of the domain name",
    "max_consonant_run": "Longest run of consecutive consonants",
    "vowel_ratio_domain": "Proportion of vowels in the domain name",
    "tld_length": "Length of the top-level domain",
    "suspicious_tld": "Top-level domain frequently abused",
    "suspicious_keywords_domain": "Security-themed keywords in the domain name",
    "has_ip": "IP address used instead of a domain name",
    "has_punycode": "Punycode (internationalised) hostname",
    "is_shortener": "URL shortening service",
    "free_hosting": "Free hosting or dynamic-DNS platform",
    "brand_in_host_mismatch": "Brand name in hostname but not the registered domain",
    "host_length": "Length of the full hostname",
    "num_dots": "Number of dots in the hostname",
    "num_subdomains": "Number of subdomains",
    "url_length": "Total URL length",
    "path_length": "Length of the path",
    "query_length": "Length of the query string",
    "num_slashes": "Number of slashes",
    "num_special_chars": "Number of special characters",
    "num_at": "Number of '@' symbols",
    "num_equals": "Number of '=' symbols",
    "num_qmark": "Number of '?' symbols",
    "num_ampersand": "Number of '&' symbols",
    "num_percent_encoded": "Number of percent-encoded characters",
    "num_digits_url": "Number of digits in the URL",
    "num_hyphens_url": "Number of hyphens in the URL",
    "url_entropy": "Randomness (Shannon entropy) of the whole URL",
    "is_https": "Uses HTTPS",
    "has_www": "Hostname starts with www",
    "has_port": "Explicit port number",
    "double_slash_in_path": "'//' redirect pattern in the path",
    "suspicious_keywords_url": "Security-themed keywords anywhere in the URL",
}

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_VALID_HOST_RE = re.compile(r"^[a-z0-9.-]+$")
_PERCENT_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")
_VOWELS = set("aeiou")

# Digits and letter pairs that are commonly swapped in to fake a brand name,
# e.g. paypa1, g00gle, rnicrosoft ("rn" looks like "m" in most fonts).
_LOOKALIKE_DIGITS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


class URLValidationError(ValueError):
    """The input can't be analysed. The message is safe to show to the user."""


@dataclass
class ParsedURL:
    original: str            # exactly what the user typed (trimmed)
    normalised: str          # with a scheme added if it was missing
    scheme: str
    host: str                # lower-case, still including any "www."
    host_no_www: str
    port: int | None
    path: str
    query: str
    registered_domain: str   # e.g. "bbc.co.uk", or the IP address itself
    subdomain: str
    tld: str
    is_ip: bool
    scheme_given: bool = True   # False when we had to add "http://" ourselves


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def parse_url(raw: str) -> ParsedURL:
    """
    Validate the user's input and split it into parts.

    Validation lives here (rather than in the Flask route) so that the
    training script, the email link extractor and the web form all reject
    exactly the same inputs.
    """
    url = "" if raw is None else str(raw).strip()
    if not url:
        raise URLValidationError("Please enter a URL.")
    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError(f"The URL is too long (maximum {MAX_URL_LENGTH} characters).")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise URLValidationError("The URL must not contain spaces or control characters.")

    # People often type "example.com" without a scheme. urlsplit() would then
    # put the whole thing in .path, so add one to get a proper hostname.
    scheme_given = bool(_SCHEME_RE.match(url))
    normalised = url if scheme_given else "http://" + url

    try:
        parts = urlsplit(normalised)
        port = parts.port          # raises ValueError for things like ":99999"
    except ValueError:
        raise URLValidationError("The URL format is not valid.") from None

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        # Also catches javascript:, data:, ftp: etc.
        raise URLValidationError("Only http:// and https:// URLs are supported.")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise URLValidationError("The URL does not contain a valid domain name.")

    is_ip = _looks_like_ip(host)
    if not is_ip:
        if not _VALID_HOST_RE.match(host) or ".." in host or host.startswith((".", "-")):
            raise URLValidationError("The domain name contains invalid characters.")
        if "." not in host:
            raise URLValidationError("The domain name must include a top-level domain (for example .com).")
        if any(len(label) > 63 for label in host.split(".")):
            raise URLValidationError("The domain name is not valid.")

    # Strip "www." only when there's still a real domain left afterwards
    # ("www.com" is a domain in its own right).
    host_no_www = host[4:] if host.startswith("www.") and host.count(".") >= 2 else host

    if is_ip:
        subdomain, registered, tld = "", host, ""
    else:
        subdomain, registered, tld = split_domain(host_no_www)

    return ParsedURL(
        original=url,
        normalised=normalised,
        scheme=scheme,
        host=host,
        host_no_www=host_no_www,
        port=port,
        path=parts.path or "",
        query=parts.query or "",
        registered_domain=registered,
        subdomain=subdomain,
        tld=tld,
        is_ip=is_ip,
        scheme_given=scheme_given,
    )


def split_domain(host: str) -> tuple[str, str, str]:
    """Split a hostname into (subdomain, registered domain, TLD)."""
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return "", host, labels[-1] if labels else ""

    last_two = ".".join(labels[-2:])
    if last_two in MULTI_PART_SUFFIXES and len(labels) >= 3:
        # news.bbc.co.uk -> ("news", "bbc.co.uk", "co.uk")
        return ".".join(labels[:-3]), ".".join(labels[-3:]), last_two

    # accounts.google.com -> ("accounts", "google.com", "com")
    return ".".join(labels[:-2]), last_two, labels[-1]


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # Browsers also accept IPv4 written as one hex or decimal number
    # (http://0x7f000001/, http://3232235777/), which phishers use to dodge
    # simple "is this an IP?" checks.
    return bool(re.fullmatch(r"0x[0-9a-f]{8}", host) or re.fullmatch(r"\d{8,10}", host))


# ---------------------------------------------------------------------------
# Small helpers shared by features and indicators
# ---------------------------------------------------------------------------

def shannon_entropy(text: str) -> float:
    """Bits per character. Randomly generated domains score noticeably higher."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in Counter(text).values())


def keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [word for word in SUSPICIOUS_KEYWORDS if word in lowered]


def impersonated_brands(p: ParsedURL) -> list[str]:
    """
    Brand names that appear in the hostname even though the site's registered
    domain isn't that brand, e.g. "paypal" in paypal.com.secure-login.xyz.
    """
    if p.is_ip or p.registered_domain in OFFICIAL_BRAND_DOMAINS:
        return []

    own_name = p.registered_domain.split(".")[0]

    # Look at everything except the public suffix. Matching against ".com" or
    # ".co.uk" would only produce noise.
    host = p.host_no_www
    if p.tld and host.endswith("." + p.tld):
        host = host[: -len(p.tld) - 1]

    # Check the host as written and with look-alike characters undone.
    candidates = {host, _undo_lookalikes(host)}

    found = []
    for brand in COMMON_BRANDS:
        if own_name == brand:
            continue  # accounts.google.com really is Google
        for text in candidates:
            tokens = [t for t in re.split(r"[.\-_]", text) if t]
            # Short brands ("ups", "ing") must be a whole token, otherwise
            # "groups" or "booking" would match. Longer ones can be embedded,
            # which catches things like "securepaypal".
            if brand in tokens or (len(brand) >= 6 and brand in text):
                found.append(brand)
                break
    return found


def _undo_lookalikes(text: str) -> str:
    return text.translate(_LOOKALIKE_DIGITS).replace("rn", "m").replace("vv", "w")


def _longest_consonant_run(text: str) -> int:
    # Real words rarely have more than 3-4 consonants in a row; keyboard-mash
    # domains like "xkqzvbt" have many.
    best = current = 0
    for ch in text:
        if ch.isalpha() and ch not in _VOWELS:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def hosting_platform(p: ParsedURL) -> str | None:
    """Return the free-hosting platform this URL sits on, if any."""
    for platform in FREE_HOSTING_DOMAINS:
        if p.host_no_www == platform or p.host_no_www.endswith("." + platform):
            return platform
    return None


def _subdomain_count(p: ParsedURL) -> int:
    return 0 if p.is_ip else len([s for s in p.subdomain.split(".") if s])


# ---------------------------------------------------------------------------
# Features for the model
# ---------------------------------------------------------------------------

def extract_features(raw_url: str, feature_set: str = "domain") -> dict[str, float]:
    """Return {feature name: value} in the exact order the model expects."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    p = parse_url(raw_url)
    domain = p.registered_domain
    # The "label" is the name part of the domain: "google" in google.com.
    # Most of the lexical signal (randomness, digits, hyphens) lives here.
    label = domain if p.is_ip else domain.split(".")[0]
    letters = [c for c in label if c.isalpha()]
    digits = sum(c.isdigit() for c in label)

    features: dict[str, float] = {
        "domain_length": len(domain),
        "domain_label_length": len(label),
        # The "xn--" prefix of punycode is not a real hyphen pattern.
        "num_hyphens_domain": label.replace("xn--", "").count("-"),
        "num_digits_domain": digits,
        "digit_ratio_domain": digits / max(len(label), 1),
        "domain_entropy": shannon_entropy(label),
        "max_consonant_run": _longest_consonant_run(label),
        "vowel_ratio_domain": sum(c in _VOWELS for c in letters) / max(len(letters), 1),
        "tld_length": len(p.tld),
        "suspicious_tld": int(bool(p.tld) and p.tld.split(".")[-1] in SUSPICIOUS_TLDS),
        "suspicious_keywords_domain": 0 if p.is_ip else len(keyword_hits(label)),
        # The red flags below look at the whole host, because that's where
        # the tricks happen (paypal.com.evil.xyz, xn--pple-43d.com, ...).
        "has_ip": int(p.is_ip),
        "has_punycode": int("xn--" in p.host_no_www),
        "is_shortener": int(p.host_no_www in URL_SHORTENERS),
        "free_hosting": int(hosting_platform(p) is not None),
        "brand_in_host_mismatch": int(bool(impersonated_brands(p))),
    }

    if feature_set == "full":
        url = p.original
        features.update({
            "host_length": len(p.host_no_www),
            "num_dots": p.host_no_www.count("."),
            "num_subdomains": _subdomain_count(p),
            "url_length": len(url),
            "path_length": len(p.path),
            "query_length": len(p.query),
            "num_slashes": url.count("/"),
            "num_special_chars": sum(not c.isalnum() for c in url),
            "num_at": url.count("@"),
            "num_equals": url.count("="),
            "num_qmark": url.count("?"),
            "num_ampersand": url.count("&"),
            "num_percent_encoded": len(_PERCENT_ENCODED_RE.findall(url)),
            "num_digits_url": sum(c.isdigit() for c in url),
            "num_hyphens_url": url.count("-"),
            "url_entropy": shannon_entropy(url),
            "is_https": int(p.scheme == "https"),
            "has_www": int(p.host.startswith("www.")),
            "has_port": int(p.port is not None),
            "double_slash_in_path": int("//" in p.path),
            "suspicious_keywords_url": len(keyword_hits(url)),
        })

    return {name: float(features[name]) for name in FEATURE_SETS[feature_set]}


def features_to_row(raw_url: str, feature_set: str = "domain") -> list[float]:
    """Same as extract_features() but as a plain list, handy for quick checks."""
    return list(extract_features(raw_url, feature_set).values())


# ---------------------------------------------------------------------------
# Rule-based warnings for the report
#
# These are deliberately kept apart from the model. They're the classic
# "things to look for" advice, and they only appear when the sign is
# actually present in the URL, so the report never shows a generic list.
# ---------------------------------------------------------------------------

def rule_based_indicators(raw_url: str) -> list[dict]:
    p = parse_url(raw_url)
    found: list[dict] = []

    def warn(title: str, detail: str, level: str = "warning") -> None:
        found.append({"title": title, "detail": detail, "level": level})

    if p.is_ip:
        warn("IP address used instead of a domain name",
             "Legitimate organisations almost always use a named domain. Raw IP addresses are a common phishing pattern.",
             "high")

    if "@" in p.original:
        warn("'@' symbol in the URL",
             "Browsers ignore everything before an '@' in the address, which can disguise the real destination.",
             "high")

    if "xn--" in p.host:
        warn("Punycode (internationalised) domain",
             "Punycode can be used to create look-alike domains that imitate well-known brands with similar-looking characters.",
             "high")

    brands = impersonated_brands(p)
    if brands:
        warn("Brand name outside the registered domain",
             f"The hostname mentions '{brands[0]}', but the registered domain is '{p.registered_domain}'. "
             "Check that the site really belongs to that organisation.",
             "high")

    if p.host_no_www in URL_SHORTENERS:
        warn("URL shortening service",
             "Short links hide the final destination. Use a link-preview service or avoid entering details.")

    platform = hosting_platform(p)
    if platform:
        warn("Hosted on a free or user-content platform",
             f"The page is hosted on '{platform}', where anyone can create a site. "
             "Many legitimate sites use these, but they are also frequently abused.")

    if p.tld and p.tld.split(".")[-1] in SUSPICIOUS_TLDS:
        warn("Frequently abused top-level domain",
             f"The '.{p.tld}' ending is often used in low-cost or disposable phishing domains.")

    subdomains = _subdomain_count(p)
    if subdomains >= 3:
        warn("Multiple subdomains",
             f"The hostname contains {subdomains} subdomain levels, which can be used to push the real domain out of view.")

    if p.host_no_www.replace("xn--", "").count("-") >= 2:
        warn("Several hyphens in the domain name",
             "Long hyphenated domains (e.g. secure-login-account-update) are common in phishing.")

    if not p.is_ip and sum(c.isdigit() for c in p.host_no_www) >= 4:
        warn("Many digits in the domain name",
             "Domains with many numbers are often automatically generated.")

    words = keyword_hits(p.original)
    if words:
        quoted = ", ".join(f"'{w}'" for w in words[:4])
        warn("Security-themed keywords",
             f"Contains words such as {quoted}. Phishing links often use these to create urgency or appear official.")

    # Only complain about HTTP if the user actually typed http:// - if they
    # typed a bare domain, the scheme is our assumption, not theirs.
    if p.scheme_given and p.scheme != "https":
        warn("No HTTPS",
             "The connection would not be encrypted. HTTPS alone does not prove a site is safe, "
             "but its absence is a warning sign.")

    if len(p.original) > 100:
        warn("Unusually long URL",
             f"The URL is {len(p.original)} characters long. Long URLs can hide the real destination.")

    if len(_PERCENT_ENCODED_RE.findall(p.original)) >= 3:
        warn("Encoded characters",
             "The URL contains several percent-encoded characters, which can be used to obscure text.")

    if "//" in p.path:
        warn("Redirect pattern in path", "A '//' inside the path can indicate a redirect to another site.")

    if p.port is not None and p.port not in (80, 443):
        warn("Non-standard port", f"The URL specifies port {p.port}. Ordinary websites rarely do this.")

    # Both thresholds are deliberately conservative heuristics: they're
    # meant to stay quiet for ordinary long domains and only fire on
    # strings that look machine-generated.
    if len(p.host_no_www) > 20 and shannon_entropy(p.host_no_www) > 4.0:
        warn("Random-looking domain name", "The domain name has an unusually random character pattern.")

    return found
