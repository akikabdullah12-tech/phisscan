"""URL parsing, validation, feature extraction and the rule-based URL warnings."""

import pytest

from ml.url_features import (FEATURE_SETS, URLValidationError, extract_features, features_to_row, parse_url,
                             rule_based_indicators)


def test_feature_order_and_count():
    row = features_to_row("https://www.example.com")
    assert len(row) == len(FEATURE_SETS["domain"])
    full = extract_features("https://www.example.com/a?b=1", "full")
    assert list(full) == FEATURE_SETS["full"]


def test_registered_domain_used_for_model_features():
    a = extract_features("https://accounts.google.com/signin")
    b = extract_features("https://google.com")
    assert a == b  # subdomains / paths do not change deployed features


def test_ip_punycode_shortener():
    assert extract_features("http://192.168.1.10/login")["has_ip"] == 1
    assert extract_features("http://xn--pple-43d.com")["has_punycode"] == 1
    assert extract_features("bit.ly/abc")["is_shortener"] == 1


def test_brand_mismatch_and_lookalikes():
    assert extract_features("http://paypal.com.secure-login.xyz")["brand_in_host_mismatch"] == 1
    assert extract_features("paypa1-secure.com")["brand_in_host_mismatch"] == 1
    assert extract_features("https://www.paypal.com")["brand_in_host_mismatch"] == 0
    assert extract_features("https://outlook.office.com")["brand_in_host_mismatch"] == 0


def test_multi_part_suffix():
    p = parse_url("https://news.bbc.co.uk/x")
    assert p.registered_domain == "bbc.co.uk" and p.tld == "co.uk" and p.subdomain == "news"


@pytest.mark.parametrize("bad", ["", "   ", "ftp://x.com", "hello world", "localhost", "javascript:alert(1)",
                                 "http://exa_mple.com", "x" * 3000, "http://", "http://a..b.com"])
def test_invalid_urls_rejected(bad):
    with pytest.raises(URLValidationError):
        parse_url(bad)


def test_scheme_optional():
    assert parse_url("example.com").host == "example.com"


def test_indicators_only_when_present():
    assert rule_based_indicators("https://www.bbc.co.uk/news") == []
    titles = [i["title"] for i in rule_based_indicators("http://192.168.0.1/login")]
    assert "IP address used instead of a domain name" in titles
    assert "No HTTPS" in titles
    # no scheme given -> do not claim missing HTTPS
    assert "No HTTPS" not in [i["title"] for i in rule_based_indicators("example.com")]
