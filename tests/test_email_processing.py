"""Email validation, cleaning, link extraction and the rule-based email warnings."""

import pytest

from ml.email_preprocessing import (EmailValidationError, clean_text, email_indicators, extract_urls, html_to_text,
                                    validate_email_input)


def test_clean_text_is_deterministic():
    assert clean_text("Visit https://x.com NOW! Call 0800 123") == "visit urltoken now call"
    assert clean_text("Mail me: a@b.com") == "mail me emailtoken"


def test_html_scripts_removed_not_executed():
    text = html_to_text("<p>Hello</p><script>alert('x')</script><style>p{}</style><b>World</b>")
    assert "alert" not in text and "Hello" in text and "World" in text


def test_extract_urls_from_text_and_href():
    body = 'Click <a href="http://bad.example.xyz/login">here</a> or visit www.example.com. mailto:x@y.com'
    urls = extract_urls(body)
    assert "http://bad.example.xyz/login" in urls
    assert "www.example.com" in urls
    assert not any(u.startswith("mailto") for u in urls)


def test_validation():
    with pytest.raises(EmailValidationError):
        validate_email_input("", "", "")
    with pytest.raises(EmailValidationError):
        validate_email_input("", "not-an-email", "hello there friend")
    with pytest.raises(EmailValidationError):
        validate_email_input("", "", "x" * 60000)
    v = validate_email_input("Hi", "Bank <alerts@bank.com>", "Hello there")
    assert v["sender"] == "alerts@bank.com" and v["sender_display"] == "Bank"


def test_indicators():
    v = validate_email_input("Urgent", "support.paypal@gmail.com",
                             'Dear customer, verify your account immediately. <a href="http://evil.xyz">www.paypal.com</a>')
    titles = {i["title"] for i in email_indicators(v, extract_urls(v["raw_body"]))}
    assert "Urgent or time-pressure language" in titles
    assert "Account verification request" in titles
    assert "Generic greeting" in titles
    assert "Link text does not match its destination" in titles
    assert "Organisation name from a free email account" in titles


def test_no_indicators_for_plain_message():
    v = validate_email_input("Lunch", "jo@example.ac.uk", "Shall we meet for lunch on Friday at noon?")
    assert email_indicators(v, []) == []
