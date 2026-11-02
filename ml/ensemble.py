"""
Soft-voting ensemble of the five algorithms.

The ensemble's phishing probability is the plain average of its members'
probabilities. An unweighted average was chosen on purpose: learning weights
would need yet another validation split, and with five reasonably strong
members the simple average is the usual starting point.

It's built from members that are already trained, so it adds no training
time of its own. It is still evaluated like every other model, though:
cross-validation estimates come from averaging the members' out-of-fold
predictions (see evaluate_models.ensemble_result), and the test score from
averaging their test-set probabilities. That way it competes in model
selection on equal terms instead of being trusted blindly.
"""

from __future__ import annotations

import numpy as np

ENSEMBLE_NAME = "Soft-Voting Ensemble"

# The five algorithms named in the project specification. Naive Bayes is
# trained for email as a comparison baseline, but it isn't one of the five,
# so it stays out of the vote.
ENSEMBLE_MEMBERS = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "XGBoost"]


class SoftVotingEnsemble:
    """Averages predict_proba over already-fitted models that take the same input."""

    def __init__(self, members: dict, threshold: float = 0.5):
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = dict(members)
        self.threshold = threshold

    def predict_proba(self, X):
        phishing = np.mean([m.predict_proba(X)[:, 1] for m in self.members.values()], axis=0)
        return np.column_stack([1 - phishing, phishing])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def all_model_verdicts(members: dict, X, selected_name: str, threshold: float = 0.5) -> tuple[list[dict], dict]:
    """
    Run one input through every algorithm, plus the ensemble.

    Returns (rows for the report table, agreement summary). The model that
    was selected for deployment is flagged so the report can highlight it.
    Only the individual algorithms count towards the agreement figure; the
    ensemble is their average, so counting it would count the same votes twice.
    """
    ordered = [name for name in ENSEMBLE_MEMBERS if name in members]
    ordered += [name for name in members if name not in ordered]

    rows = []
    for name in ordered:
        probability = float(members[name].predict_proba(X)[0, 1])
        rows.append({"model": name, "probability": probability, "is_phishing": probability >= threshold,
                     "selected": name == selected_name, "is_ensemble": False})

    phishing_votes = sum(r["is_phishing"] for r in rows)
    agreement = {"phishing": phishing_votes, "legitimate": len(rows) - phishing_votes, "total": len(rows)}

    if len(rows) >= 2:
        average = float(np.mean([r["probability"] for r in rows]))
        rows.append({"model": ENSEMBLE_NAME, "probability": average, "is_phishing": average >= threshold,
                     "selected": selected_name == ENSEMBLE_NAME, "is_ensemble": True})
    return rows, agreement
