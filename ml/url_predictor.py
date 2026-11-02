"""
Runs the saved URL model on a single URL.

The URL is only ever handled as text: it's validated, turned into the same
features used in training, and passed to the model. The site is not visited.
"""

from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np

from ml.ensemble import all_model_verdicts
from ml.url_features import FEATURE_DESCRIPTIONS, extract_features, parse_url, rule_based_indicators

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "url_model.joblib"


class ModelNotReadyError(RuntimeError):
    """No usable trained model. The web app shows this as 'detection not ready'
    rather than ever falling back to a made-up result."""


class URLPredictor:
    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        self._bundle = None
        self._loaded_mtime = None
        # Flask's dev server is threaded, so two first requests could
        # otherwise both try to load the (40 MB) model at the same moment.
        self._lock = threading.Lock()

    def _load(self) -> dict:
        """Load lazily, and pick up a retrained model without restarting the app."""
        if not self.path.exists():
            raise ModelNotReadyError(
                "The URL detection model has not been trained yet. Run: python -m ml.train_url_models")

        mtime = self.path.stat().st_mtime
        with self._lock:
            if self._bundle is None or mtime != self._loaded_mtime:
                try:
                    self._bundle = joblib.load(self.path)
                except Exception as exc:
                    # Usually a scikit-learn version mismatch after an upgrade.
                    raise ModelNotReadyError("The URL detection model could not be loaded. Please retrain it.") from exc
                self._loaded_mtime = mtime
        return self._bundle

    def is_ready(self) -> bool:
        try:
            self._load()
            return True
        except ModelNotReadyError:
            return False

    def info(self) -> dict | None:
        try:
            bundle = self._load()
        except ModelNotReadyError:
            return None
        return {k: bundle.get(k) for k in ("model_name", "model_version", "feature_set", "trained_at",
                                           "threshold", "test_metrics", "quick_mode")}

    def predict(self, raw_url: str) -> dict:
        # Validate before loading anything, so bad input gets a helpful
        # message even if the model happens to be missing.
        parsed = parse_url(raw_url)
        bundle = self._load()

        names = bundle["feature_names"]
        features = extract_features(raw_url, bundle["feature_set"])
        x = np.array([[features[n] for n in names]], dtype=float)

        model = bundle["pipeline"]
        threshold = bundle.get("threshold", 0.5)
        probability = float(model.predict_proba(x)[0, 1])
        is_phishing = probability >= threshold

        # Every algorithm sees the same features, so the report can show
        # where they agree and where they don't. The headline verdict above
        # still comes only from the selected model.
        all_models, agreement = all_model_verdicts(bundle.get("members", {}), x, bundle["model_name"], threshold)

        return {
            "url": parsed.original,
            "hostname": parsed.host,
            "registered_domain": parsed.registered_domain,
            "prediction": "Phishing" if is_phishing else "Legitimate",
            "is_phishing": bool(is_phishing),
            "probability": probability,
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "feature_set": bundle["feature_set"],
            "quick_mode": bundle.get("quick_mode", False),
            "all_models": all_models,
            "agreement": agreement,
            "model_explanation": _explain(model, x[0], names, bundle["reference_values"], probability),
            "rule_indicators": rule_based_indicators(raw_url),
            "features": dict(zip(names, x[0].tolist())),
        }


def _explain(model, x, names, reference, probability, top: int = 5) -> list[dict]:
    """
    A simple "what if this looked normal?" explanation.

    Each feature is swapped, one at a time, for its typical value among
    legitimate training domains (the stored medians), and the change in the
    phishing probability is recorded. A positive effect means that feature's
    real value was pushing the prediction towards phishing.

    I went with this over SHAP to avoid another heavy dependency. It works
    the same way for every model type and only costs one extra batch
    prediction. It explains the model's behaviour; it says nothing about
    the website's intent.
    """
    what_ifs = np.repeat(x[None, :], len(names), axis=0)
    for i, typical in enumerate(reference):
        what_ifs[i, i] = typical
    changed = model.predict_proba(what_ifs)[:, 1]

    effects = []
    for i, name in enumerate(names):
        effect = probability - float(changed[i])
        # Skip features that already have the typical value, and ones whose
        # effect is too small to be worth a line in the report.
        if x[i] == reference[i] or abs(effect) < 0.01:
            continue
        effects.append({
            "feature": name,
            "description": FEATURE_DESCRIPTIONS.get(name, name),
            "value": float(x[i]),
            "typical_legitimate_value": float(reference[i]),
            "effect": effect,
        })

    effects.sort(key=lambda e: abs(e["effect"]), reverse=True)
    return effects[:top]


# Shared instance used by the web app; loads lazily on first use.
predictor = URLPredictor()
