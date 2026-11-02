"""
Shared training/evaluation code for the URL and email experiments.

Both training scripts follow the same recipe, so the parts they have in
common live here: the candidate models and their search grids, the
tune -> refit -> test routine, the model-selection rule, and writing the
results (JSON, CSV and charts) that the admin page and the dissertation
tables are built from.

Running this module directly prints the latest stored results:

    python -m ml.evaluate_models
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # charts are written to files; there's no display on a server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# XGBoost is in requirements.txt, but I don't want a missing or broken
# install to stop the other four models from training. If it can't be
# imported, it's skipped and the results file records that it was skipped.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the machine
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "research_outputs" / "model_evaluation"
MODELS_DIR = ROOT / "saved_models"

# One seed everywhere so the split, the CV folds and the models are
# reproducible between runs.
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def false_positive_rate(y_true, y_pred) -> float:
    """Share of legitimate items wrongly flagged as phishing."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


# Scored during cross-validation alongside F1. A tool that cries wolf gets
# ignored, so the false-positive rate is part of how a model gets chosen.
# scikit-learn maximises scores, hence greater_is_better=False (it stores
# the value negated; tune_and_evaluate flips it back).
CV_SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "fpr": make_scorer(false_positive_rate, greater_is_better=False),
}


def test_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)

    # numpy floats don't serialise to JSON, so convert them here once.
    return {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}


def phishing_scores(model, X):
    """Continuous scores for ROC/PR curves (probabilities where available)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def candidate_models(kind: str, include_nb: bool = False, quick: bool = False) -> dict:
    """
    Return {display name: (estimator, parameter grid)} for "url" or "email".

    The grids are small on purpose. Each grid point is fitted cv_folds
    times, and on a laptop the email Random Forest and XGBoost runs are
    already the slowest part of the project. They cover the settings that
    usually matter most for each algorithm (regularisation strength, tree
    depth, leaf size, learning rate). --quick shrinks everything to one
    setting so the whole pipeline can be smoke-tested in seconds.
    """
    n_trees = 100 if quick else 200

    def grid(full: dict, quick_version: dict) -> dict:
        return quick_version if quick else full

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            grid({"C": [0.1, 1.0, 10.0]}, {"C": [1.0]}),
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            grid({"max_depth": [10, 20, None], "min_samples_leaf": [1, 5]} if kind == "url"
                 else {"max_depth": [50, None], "min_samples_leaf": [1, 3]},
                 {"max_depth": [20]}),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE),
            grid({"max_depth": [None, 25], "min_samples_leaf": [1, 3]} if kind == "url"
                 else {"max_depth": [None], "min_samples_leaf": [1, 2]},
                 {"max_depth": [None]}),
        ),
        # A kernel SVM is out of the question at ~175k training rows, so this
        # is a linear SVM. LinearSVC has no predict_proba, and the report
        # needs a probability, so it's wrapped in Platt-style calibration.
        "Support Vector Machine": (
            CalibratedClassifierCV(LinearSVC(dual="auto", max_iter=5000, random_state=RANDOM_STATE), cv=3),
            grid({"estimator__C": [0.1, 1.0, 10.0] if kind == "url" else [0.1, 1.0]},
                 {"estimator__C": [1.0]}),
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=n_trees if kind == "url" else 150,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            grid({"max_depth": [4, 8], "learning_rate": [0.1, 0.3]} if kind == "url"
                 else {"max_depth": [6], "learning_rate": [0.1, 0.3]},
                 {"max_depth": [6]}),
        )

    if include_nb:
        # Naive Bayes is the traditional text-classification baseline, so it's
        # useful as a reference point for the email models.
        models["Multinomial Naive Bayes"] = (MultinomialNB(), {"alpha": [0.05, 0.1, 0.5, 1.0]})

    return models


# ---------------------------------------------------------------------------
# Tuning, evaluation and selection
# ---------------------------------------------------------------------------

def tune_and_evaluate(name, pipeline, param_grid, X_tune, y_tune, X_train, y_train, X_test, y_test,
                      cv_folds: int = 3):
    """
    Pick hyperparameters by cross-validation, refit, then test once.

    1. GridSearchCV over (X_tune, y_tune). That's the training set, or a
       stratified sample of it when the full set would make the search too
       slow. The test set is never seen here.
    2. The winning settings are refitted on the full training set.
    3. That model is scored once on the held-out test set.

    Returns the fitted model and a results dict. Two arrays ride along for
    later use and are stripped before anything is saved: "_scores" (test-set
    probabilities, for the curves and the ensemble) and "_oof" (out-of-fold
    probabilities on the tuning data, for the ensemble's CV estimate).
    """
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, param_grid, scoring=CV_SCORING, refit="f1", cv=folds,
                          n_jobs=1, error_score="raise")

    started = time.time()
    search.fit(X_tune, y_tune)
    tuning_seconds = time.time() - started

    best = search.best_index_
    cv = search.cv_results_
    cv_summary = {
        "cv_f1_mean": float(cv["mean_test_f1"][best]),
        "cv_f1_std": float(cv["std_test_f1"][best]),
        "cv_recall_mean": float(cv["mean_test_recall"][best]),
        "cv_precision_mean": float(cv["mean_test_precision"][best]),
        # stored negated by sklearn (see CV_SCORING); max() avoids printing "-0.000"
        "cv_fpr_mean": max(0.0, float(-cv["mean_test_fpr"][best])),
    }

    # Out-of-fold probabilities for the winning settings, using the same folds.
    # GridSearchCV doesn't keep these, and the ensemble's cross-validation
    # estimate is built from them (see ensemble_result).
    oof = cross_val_predict(clone(pipeline).set_params(**search.best_params_), X_tune, y_tune,
                            cv=folds, method="predict_proba")[:, 1]

    model = clone(pipeline).set_params(**search.best_params_)
    started = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - started

    scores = phishing_scores(model, X_test)
    result = {
        "model": name,
        "best_params": {k: v if isinstance(v, (int, float, str, type(None))) else str(v)
                        for k, v in search.best_params_.items()},
        **cv_summary,
        **test_metrics(y_test, model.predict(X_test), scores),
        "tuning_seconds": round(tuning_seconds, 2),
        "final_fit_seconds": round(fit_seconds, 2),
        "_scores": scores,
        "_oof": oof,
    }
    return model, result


def ensemble_result(name: str, member_results: list[dict], y_tune, y_test, cv_folds: int = 3) -> dict:
    """
    Evaluate the soft-voting ensemble the same way as the single models.

    CV figures: average the members' out-of-fold probabilities and score each
    fold separately (the folds are identical to the ones used for tuning, as
    they come from the same seeded StratifiedKFold). Test figures: average
    the members' test-set probabilities. No member is refitted, and the test
    set plays no part in building the ensemble.
    """
    y_tune = np.asarray(y_tune)
    oof = np.mean([r["_oof"] for r in member_results], axis=0)
    test_scores = np.mean([r["_scores"] for r in member_results], axis=0)

    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    per_fold = {"f1": [], "recall": [], "precision": [], "fpr": []}
    for _, val_idx in folds.split(np.zeros(len(y_tune)), y_tune):
        truth, pred = y_tune[val_idx], (oof[val_idx] >= 0.5).astype(int)
        per_fold["f1"].append(f1_score(truth, pred, zero_division=0))
        per_fold["recall"].append(recall_score(truth, pred, zero_division=0))
        per_fold["precision"].append(precision_score(truth, pred, zero_division=0))
        per_fold["fpr"].append(false_positive_rate(truth, pred))

    return {
        "model": name,
        "best_params": {"members": [r["model"] for r in member_results], "voting": "soft (mean probability)"},
        "cv_f1_mean": float(np.mean(per_fold["f1"])),
        "cv_f1_std": float(np.std(per_fold["f1"])),
        "cv_recall_mean": float(np.mean(per_fold["recall"])),
        "cv_precision_mean": float(np.mean(per_fold["precision"])),
        "cv_fpr_mean": float(np.mean(per_fold["fpr"])),
        **test_metrics(y_test, (test_scores >= 0.5).astype(int), test_scores),
        "tuning_seconds": 0.0,      # no tuning of its own: it reuses the tuned members
        "final_fit_seconds": 0.0,
        "_scores": test_scores,
    }


def select_model(results: list[dict], max_fpr: float) -> tuple[dict, str]:
    """
    Choose the model to deploy, using cross-validation results only.

    Rule: among models whose mean CV false-positive rate is at or below
    max_fpr, take the highest mean CV F1, using CV recall as the tie-break.
    If nothing meets the FPR limit, fall back to the lowest FPR.

    Selecting on the test set would quietly turn it into a validation set
    and make the reported test scores optimistic, which is why it isn't used.
    """
    eligible = [r for r in results if r["cv_fpr_mean"] <= max_fpr]
    if eligible:
        chosen = max(eligible, key=lambda r: (r["cv_f1_mean"], r["cv_recall_mean"]))
        rule = (f"Highest mean cross-validation F1 among models with mean CV false-positive rate "
                f"<= {max_fpr:.0%} (ties broken by CV phishing recall). "
                "Test-set results were not used for selection.")
    else:
        chosen = min(results, key=lambda r: r["cv_fpr_mean"])
        rule = (f"No model met the CV false-positive-rate limit of {max_fpr:.0%}; "
                "the model with the lowest CV false-positive rate was selected.")
    return chosen, rule


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_plots(prefix: str, results: list[dict], y_test) -> list[str]:
    """Write confusion matrices, ROC curves and PR curves; return the file names."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        _plot_confusion_matrices(prefix, results),
        _plot_curves(prefix, results, y_test, kind="roc"),
        _plot_curves(prefix, results, y_test, kind="pr"),
    ]
    return written


def _plot_confusion_matrices(prefix: str, results: list[dict]) -> str:
    cols = min(len(results), 3)
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes, results):
        cm = np.array(result["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (row, col), value in np.ndenumerate(cm):
            colour = "white" if value > cm.max() / 2 else "black"
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=colour, fontsize=11)
        ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
        ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(result["model"], fontsize=11)

    for ax in axes[len(results):]:
        ax.axis("off")   # hide the empty slots in the last row

    fig.tight_layout()
    path = EVAL_DIR / f"{prefix}_confusion_matrices.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_curves(prefix: str, results: list[dict], y_test, kind: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        scores = result.get("_scores")
        if scores is None:
            continue
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_test, scores)
            ax.plot(fpr, tpr, label=f"{result['model']} (AUC {result['roc_auc']:.3f})")
        else:
            precision, recall, _ = precision_recall_curve(y_test, scores)
            ax.plot(recall, precision, label=f"{result['model']} (AP {result['pr_auc']:.3f})")

    if kind == "roc":
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)   # chance line
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (phishing recall)")
        ax.set_title("ROC curves (test set)")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curves (test set)")
        ax.legend(fontsize=8, loc="lower left")

    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = EVAL_DIR / f"{prefix}_{kind}_curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def save_results(prefix: str, run_info: dict, results: list[dict]) -> None:
    """Save the full run as JSON (read by the admin page) and a CSV for spreadsheets."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]

    (EVAL_DIR / f"{prefix}_results.json").write_text(json.dumps({**run_info, "models": public}, indent=2))

    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
               "false_negative_rate", "cv_f1_mean", "cv_f1_std", "cv_recall_mean", "cv_precision_mean",
               "cv_fpr_mean", "tuning_seconds", "final_fit_seconds", "best_params"]
    with open(EVAL_DIR / f"{prefix}_model_comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in public:
            writer.writerow({**row, "best_params": json.dumps(row.get("best_params", {}))})


def load_results(prefix: str) -> dict | None:
    path = EVAL_DIR / f"{prefix}_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None   # a half-written file shouldn't take the admin page down


def print_table(results: list[dict], selected: str | None = None) -> None:
    header = f"{'Model':<26}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}{'FPR':>7}{'CV-F1':>8}{'CV-FPR':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        marker = "  <- selected" if r["model"] == selected else ""
        print(f"{r['model']:<26}{r['accuracy']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r.get('roc_auc', float('nan')):>7.3f}{r['false_positive_rate']:>7.3f}"
              f"{r['cv_f1_mean']:>8.3f}{r['cv_fpr_mean']:>8.3f}{marker}")


def main() -> None:
    runs = [
        ("url", "URL models (deployed feature set)"),
        ("url_full", "URL models - full-URL feature experiment"),
        ("email", "Email models"),
    ]
    for prefix, title in runs:
        print(f"\n=== {title} ===")
        data = load_results(prefix)
        if not data:
            print("No results yet. Run the training script first.")
            continue
        rows = data.get("dataset", {}).get("rows_after_cleaning")
        print(f"Trained: {data.get('trained_at')}  |  Dataset rows after cleaning: {rows}")
        print_table(data["models"], data.get("selected_model"))
        print("Selection rule:", data.get("selection_rule"))
        if data.get("skipped_models"):
            print("Skipped:", data["skipped_models"])


if __name__ == "__main__":
    main()
