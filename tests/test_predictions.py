"""
Model predictions. These need trained models in saved_models/, and are
skipped (not failed) if the models haven't been trained yet.
"""

import pytest

from ml.email_predictor import EmailPredictor
from ml.url_features import URLValidationError
from ml.url_predictor import ModelNotReadyError, URLPredictor, predictor as url_predictor
from ml.email_predictor import predictor as email_predictor
from tests.conftest import requires_email_model, requires_url_model


def test_missing_model_reports_not_ready(tmp_path):
    p = URLPredictor(tmp_path / "missing.joblib")
    assert not p.is_ready()
    with pytest.raises(ModelNotReadyError):
        p.predict("https://example.com")
    e = EmailPredictor(tmp_path / "m.joblib", tmp_path / "v.joblib")
    with pytest.raises(ModelNotReadyError):
        e.predict("", "", "hello there")


@requires_url_model
def test_url_prediction_is_deterministic_and_bounded():
    a = url_predictor.predict("http://paypal-secure-login.xyz/verify")
    b = url_predictor.predict("http://paypal-secure-login.xyz/verify")
    assert a["probability"] == b["probability"]
    assert 0.0 <= a["probability"] <= 1.0
    assert a["prediction"] in ("Phishing", "Legitimate")
    assert a["model_version"]


@requires_url_model
def test_url_obvious_cases():
    assert url_predictor.predict("http://paypal-secure-login-verify.xyz")["is_phishing"]
    assert not url_predictor.predict("https://www.bbc.co.uk")["is_phishing"]


@requires_url_model
def test_url_invalid_input():
    with pytest.raises(URLValidationError):
        url_predictor.predict("not a url")


@requires_email_model
def test_email_prediction_and_embedded_urls():
    r = email_predictor.predict(
        "Your account will be suspended", "",
        "Dear customer, we detected unusual activity. Verify your account and confirm your password within 24 hours "
        "at http://secure-login-verify.xyz/account or it will be closed.")
    assert r["is_phishing"]
    assert len(r["urls"]) == 1 and r["urls"][0]["prediction"] in ("Phishing", "Legitimate")
    legit = email_predictor.predict("Meeting notes", "jo@example.ac.uk",
                                    "Hi all, the notes from Tuesday's meeting are below. Next meeting Thursday at 2pm. Thanks, Jo")
    assert not legit["is_phishing"]


@requires_url_model
def test_url_report_data_includes_every_algorithm():
    r = url_predictor.predict("http://paypal-secure-login-verify.xyz")
    individual = [m for m in r["all_models"] if not m["is_ensemble"]]
    assert len(individual) >= 4                      # 5 once XGBoost is installed
    assert r["agreement"]["total"] == len(individual)
    assert sum(m["selected"] for m in r["all_models"]) == 1
    selected = next(m for m in r["all_models"] if m["selected"])
    assert selected["model"] == r["model_name"]
