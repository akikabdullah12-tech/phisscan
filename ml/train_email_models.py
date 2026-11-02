"""
Train and evaluate the email classifiers (TF-IDF text features).

Usage (from the project folder):

    python -m ml.train_email_models
    python -m ml.train_email_models --quick      # 5k-row smoke test

Models: Logistic Regression, Decision Tree, Random Forest, linear SVM,
XGBoost, plus Multinomial Naive Bayes as the classic text baseline.

The TF-IDF vectoriser is the first step of each Pipeline, so inside every
cross-validation fold it only learns its vocabulary and IDF weights from
that fold's training part. Fitting it on the whole dataset first would
leak information from the test data into training.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml import evaluate_models as ev
from ml.ensemble import ENSEMBLE_MEMBERS, ENSEMBLE_NAME, SoftVotingEnsemble
from ml.email_predictor import term_weights
from ml.email_preprocessing import clean_text

DEFAULT_DATA = ev.ROOT / "data" / "Phishing_Email.csv"

# Column names and label spellings differ between the public email
# datasets, so accept the common ones instead of hard-coding a single file.
TEXT_COLUMNS = ["Email Text", "text", "body", "Text", "email"]
LABEL_COLUMNS = ["Email Type", "label", "Label", "class"]
PHISHING_LABELS = {"phishing email", "phishing", "spam", "1", "malicious"}
LEGITIMATE_LABELS = {"safe email", "legitimate", "ham", "safe", "0", "benign"}


def make_vectorizer(quick: bool = False) -> TfidfVectorizer:
    # Reasoning behind the settings:
    #  - unigrams + bigrams capture phrases like "verify account" and "click here"
    #  - min_df=2 drops one-off typos and IDs; max_df=0.95 drops near-universal words
    #  - sublinear_tf stops one long, repetitive email dominating its own vector
    #  - a 20k-term cap keeps the matrix (and the tree models) a manageable size
    return TfidfVectorizer(
        max_features=5000 if quick else 20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,   # half the memory of float64, and no measurable difference in accuracy
    )


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nSee data/README.md for how to obtain it.")

    df = pd.read_csv(path, encoding="utf-8-sig")
    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
    if not text_col or not label_col:
        sys.exit(f"Could not find text/label columns in {path.name}. Columns: {list(df.columns)}")

    stats = {"rows_loaded": int(len(df))}
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(PHISHING_LABELS | LEGITIMATE_LABELS)]
    df["target"] = df["label"].isin(PHISHING_LABELS).astype(int)

    # Clean first, then de-duplicate on the cleaned text. Many of the
    # duplicates in this dataset differ only in whitespace or numbers, and
    # a duplicate split across train and test would inflate the scores.
    df["clean"] = df["text"].astype(str).map(clean_text)
    df = df[df["clean"].str.len() >= 3]
    stats["after_dropping_empty"] = int(len(df))

    labels_per_text = df.groupby("clean")["target"].nunique()
    contradictory = set(labels_per_text[labels_per_text > 1].index)
    df = df[~df["clean"].isin(contradictory)]
    stats["conflicting_texts_dropped"] = len(contradictory)

    df = df.drop_duplicates(subset=["clean"]).reset_index(drop=True)
    stats["rows_after_cleaning"] = int(len(df))
    stats["phishing_rows"] = int(df["target"].sum())
    stats["legitimate_rows"] = int((df["target"] == 0).sum())
    return df, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the PhisScan email models")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--no-naive-bayes", action="store_true")
    parser.add_argument("--quick", action="store_true", help="small grids and a 5k-row sample")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.time()

    print(f"Loading {args.data} ...")
    df, stats = load_dataset(args.data)
    if args.quick:
        df = df.sample(n=min(5000, len(df)), random_state=ev.RANDOM_STATE)
    print(f"  {stats}")

    X = df["clean"].to_numpy()
    y = df["target"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=ev.RANDOM_STATE)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    results, fitted = [], {}
    skipped = [] if ev.XGBOOST_AVAILABLE else ["XGBoost (package not installed: pip install xgboost)"]
    candidates = ev.candidate_models("email", include_nb=not args.no_naive_bayes, quick=args.quick)

    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([("tfidf", make_vectorizer(args.quick)), ("clf", estimator)])
        grid = {f"clf__{param}": values for param, values in grid.items()}

        print(f"\nTraining {name} ...")
        # The email set is small enough (~14k training rows) to tune on all of it.
        model, result = ev.tune_and_evaluate(name, pipeline, grid, X_train, y_train, X_train, y_train,
                                             X_test, y_test, cv_folds=args.cv_folds)
        total = result["tuning_seconds"] + result["final_fit_seconds"]
        print(f"  CV F1 {result['cv_f1_mean']:.4f} | test F1 {result['f1']:.4f} | "
              f"test FPR {result['false_positive_rate']:.4f} | {total:.0f}s")
        results.append(result)
        fitted[name] = model

    # Every pipeline's TF-IDF step was fitted on the same training texts with
    # the same settings, so their vocabularies are identical. That lets the
    # app transform an email once and hand the result to every classifier.
    # The assert is there in case someone changes make_vectorizer per model.
    vectorizer = next(iter(fitted.values())).named_steps["tfidf"]
    for pipeline in fitted.values():
        assert pipeline.named_steps["tfidf"].vocabulary_ == vectorizer.vocabulary_
    classifiers = {name: pipeline.named_steps["clf"] for name, pipeline in fitted.items()}

    members = [r for r in results if r["model"] in ENSEMBLE_MEMBERS]
    if len(members) >= 2:
        print(f"\nEvaluating {ENSEMBLE_NAME} of {len(members)} models ...")
        results.append(ev.ensemble_result(ENSEMBLE_NAME, members, y_train, y_test, args.cv_folds))
        classifiers[ENSEMBLE_NAME] = SoftVotingEnsemble({r["model"]: classifiers[r["model"]] for r in members})

    chosen, rule = ev.select_model(results, args.max_fpr)
    print()
    ev.print_table(results, chosen["model"])
    print("\nSelection rule:", rule)

    classifier = classifiers[chosen["model"]]
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    version = f"email-{chosen['model'].lower().replace(' ', '_')}-{datetime.now():%Y%m%d%H%M}"

    ev.save_results("email", {
        "task": "email",
        "dataset": {"file": args.data.name, **stats},
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "cv_folds": args.cv_folds,
                  "test_size": args.test_size, "random_state": ev.RANDOM_STATE},
        "vectorizer": {
            "type": "TF-IDF",
            "max_features": vectorizer.max_features,
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "sublinear_tf": vectorizer.sublinear_tf,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "selected_model": chosen["model"],
        "selection_rule": rule,
        "model_version": version,
        "trained_at": trained_at,
        "skipped_models": skipped,
        "top_phishing_terms": _top_phishing_terms(vectorizer, classifier),
        "plots": ev.save_plots("email", results, y_test),
        "quick_mode": bool(args.quick),
        "sklearn_version": sklearn.__version__,
    }, results)

    # The vectoriser and classifier are saved as separate files so the
    # vocabulary can be inspected on its own.
    ev.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, ev.MODELS_DIR / "email_vectorizer.joblib", compress=3)
    joblib.dump({
        "classifier": classifier,
        # Every individual algorithm, for the side-by-side table in the report.
        # Naive Bayes is left out: it's a training baseline, not one of the five.
        "members": {r["model"]: classifiers[r["model"]] for r in members},
        "model_name": chosen["model"],
        "model_version": version,
        "threshold": 0.5,
        "trained_at": trained_at,
        "test_metrics": {k: chosen[k] for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate")},
        "quick_mode": bool(args.quick),
        "label_map": {0: "Legitimate", 1: "Phishing"},
    }, ev.MODELS_DIR / "email_model.joblib", compress=3)

    print(f"\nSaved deployed model: saved_models/email_model.joblib + email_vectorizer.joblib ({chosen['model']})")
    print(f"Results: research_outputs/model_evaluation/email_*  |  total time {time.time() - started:.0f}s")


def _top_phishing_terms(vectorizer, classifier, n: int = 25) -> list[dict]:
    """The terms the chosen model links most strongly to phishing (for the admin page)."""
    weights, signed = term_weights(classifier)
    if weights is None:
        return []
    vocabulary = vectorizer.get_feature_names_out()
    top = np.argsort(weights)[::-1][:n]
    return [{"term": str(vocabulary[i]), "weight": float(weights[i]), "signed": signed} for i in top]


if __name__ == "__main__":
    main()
