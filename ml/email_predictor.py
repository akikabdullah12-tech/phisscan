"""
Runs the saved email model on a pasted email.

For each submission this:
  1. validates and cleans the input (email_preprocessing),
  2. scores the text with the saved TF-IDF vectoriser and classifier,
  3. pulls out any links and runs each one through the URL model,
  4. collects the rule-based warnings,
and returns everything the email report page needs.

Nothing in the email is rendered or executed, and no link is ever opened.
"""

from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np

from ml.ensemble import all_model_verdicts
from ml.email_preprocessing import (
    EmailValidationError,  # noqa: F401  (re-exported for the Flask routes)
    email_indicators,
    extract_urls,
    model_input,
    validate_email_input,
)
from ml.url_features import URLValidationError
from ml.url_predictor import ModelNotReadyError, predictor as url_predictor

MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_PATH = MODELS_DIR / "email_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "email_vectorizer.joblib"


def term_weights(classifier):
    """
    Per-term weights for explaining a prediction.

    Returns (weights, signed). With signed weights (linear models and Naive
    Bayes), a positive value pushes towards phishing. Tree models only give
    unsigned importances, so they can say a word mattered but not which way.
    The training script uses this too, for the "top phishing terms" list.
    """
    # For the ensemble, average the members that have signed weights (the
    # linear ones). Each is scaled to a max of 1 first, because their raw
    # coefficients are on very different scales.
    if hasattr(classifier, "members"):
        signed_weights = []
        for member in classifier.members.values():
            weights, signed = term_weights(member)
            if weights is not None and signed and np.abs(weights).max() > 0:
                signed_weights.append(weights / np.abs(weights).max())
        if signed_weights:
            return np.mean(signed_weights, axis=0), True
        return None, False

    # The calibrated SVM wraps several LinearSVCs (one per calibration fold),
    # so average their coefficients.
    if hasattr(classifier, "calibrated_classifiers_"):
        coefs = [c.estimator.coef_[0] for c in classifier.calibrated_classifiers_ if hasattr(c.estimator, "coef_")]
        if coefs:
            return np.mean(coefs, axis=0), True
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_).ravel(), True
    if hasattr(classifier, "feature_log_prob_"):
        # Naive Bayes: how much more likely the word is under "phishing".
        return classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0], True
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_), False
    return None, False


class EmailPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self._loaded = None
        self._loaded_mtimes = None
        self._lock = threading.Lock()

    def _load(self):
        """Load the model files, and reload them if training has replaced them."""
        if not (self.model_path.exists() and self.vectorizer_path.exists()):
            raise ModelNotReadyError(
                "The email detection model has not been trained yet. Run: python -m ml.train_email_models")

        mtimes = (self.model_path.stat().st_mtime, self.vectorizer_path.stat().st_mtime)
        with self._lock:
            if self._loaded is None or mtimes != self._loaded_mtimes:
                try:
                    bundle = joblib.load(self.model_path)
                    vectorizer = joblib.load(self.vectorizer_path)
                except Exception as exc:
                    raise ModelNotReadyError(
                        "The email detection model could not be loaded. Please retrain it.") from exc
                weights, signed = term_weights(bundle["classifier"])
                # Work out the vocabulary and weights once at load time,
                # not on every request.
                self._loaded = (bundle, vectorizer, weights, signed, vectorizer.get_feature_names_out())
                self._loaded_mtimes = mtimes
        return self._loaded

    def is_ready(self) -> bool:
        try:
            self._load()
            return True
        except ModelNotReadyError:
            return False

    def info(self) -> dict | None:
        try:
            bundle = self._load()[0]
        except ModelNotReadyError:
            return None
        return {k: bundle.get(k) for k in ("model_name", "model_version", "trained_at", "threshold",
                                           "test_metrics", "quick_mode")}

    def predict(self, subject: str | None, sender: str | None, body: str | None) -> dict:
        email = validate_email_input(subject, sender, body)
        bundle, vectorizer, weights, signed, vocabulary = self._load()

        # Transform once; every classifier was trained on this same vocabulary.
        X = vectorizer.transform([model_input(email["subject"], email["body_text"])])
        threshold = bundle.get("threshold", 0.5)
        probability = float(bundle["classifier"].predict_proba(X)[0, 1])
        is_phishing = probability >= threshold
        all_models, agreement = all_model_verdicts(bundle.get("members", {}), X, bundle["model_name"], threshold)

        urls = extract_urls(email["raw_body"])

        return {
            "subject": email["subject"],
            "sender": email["sender"],
            "prediction": "Phishing" if is_phishing else "Legitimate",
            "is_phishing": bool(is_phishing),
            "probability": probability,
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "quick_mode": bundle.get("quick_mode", False),
            "all_models": all_models,
            "agreement": agreement,
            "model_explanation": _influential_terms(X, weights, signed, vocabulary),
            "explanation_signed": signed,
            "rule_indicators": email_indicators(email, urls),
            "urls": [_check_link(url) for url in urls],
            "word_count": len(email["body_text"].split()),
        }


def _check_link(url: str) -> dict:
    """Run one embedded link through the URL model. A bad link shouldn't fail the whole email."""
    result = {"url": url}
    try:
        r = url_predictor.predict(url)
    except URLValidationError as exc:
        result.update(prediction="Not analysed", error=str(exc))
        return result
    except ModelNotReadyError:
        result.update(prediction="Not analysed", error="URL model not available")
        return result

    result.update(
        agreement=r["agreement"],
        prediction=r["prediction"],
        is_phishing=r["is_phishing"],
        probability=r["probability"],
        registered_domain=r["registered_domain"],
        indicators=[i["title"] for i in r["rule_indicators"]],
    )
    return result


# Placeholder tokens from clean_text() would look odd in the report.
_FRIENDLY_TOKENS = {"urltoken": "[link]", "emailtoken": "[email address]"}


def _influential_terms(X, weights, signed, vocabulary, top: int = 8) -> list[dict]:
    """
    The words in this email that contributed most to its score.

    For signed models the contribution is weight x TF-IDF value, and only
    positive (towards phishing) terms are listed. For tree models it's
    importance x TF-IDF value, and the report says the direction is unknown.
    """
    if weights is None:
        return []
    row = X.tocsr()
    if row.nnz == 0:
        return []

    contributions = weights[row.indices] * row.data
    order = np.argsort(contributions if signed else np.abs(contributions))[::-1]

    terms = []
    for i in order[:top]:
        value = float(contributions[i])
        if signed and value <= 0:
            break
        term = str(vocabulary[row.indices[i]])
        for token, label in _FRIENDLY_TOKENS.items():
            term = term.replace(token, label)
        terms.append({"term": term, "contribution": value})
    return terms


# Shared instance used by the web app; loads lazily on first use.
predictor = EmailPredictor()
