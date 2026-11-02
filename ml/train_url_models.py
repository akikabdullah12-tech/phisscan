"""
Train and evaluate the URL classifiers on PhiUSIIL.

Usage (from the project folder):

    python -m ml.train_url_models                     # train + deploy the domain-feature model
    python -m ml.train_url_models --quick             # quick smoke test on a 30k sample
    python -m ml.train_url_models --feature-set full  # the artefact experiment (not deployed)

Steps: load the CSV, flip the labels, drop duplicates and contradictory rows,
extract features, make a stratified 80/20 split, tune each model with
cross-validation on the training data, test once, pick a model from the CV
results, then save the model and the results files.

PhiUSIIL uses 1 = legitimate. Everything in PhisScan uses 1 = phishing,
so the labels are flipped as soon as they're loaded.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import evaluate_models as ev
from ml.ensemble import ENSEMBLE_MEMBERS, ENSEMBLE_NAME, SoftVotingEnsemble
from ml.url_features import FEATURE_SETS, URLValidationError, extract_features, parse_url

DEFAULT_DATA = ev.ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"

# Linear models are sensitive to feature scale (URL length is in the
# hundreds, ratios are 0-1), so these two get a StandardScaler in front.
# Tree models don't care about scale.
NEEDS_SCALING = {"Logistic Regression", "Support Vector Machine"}


def load_dataset(path: Path, label_column: str = "label", legitimate_value: int = 1) -> pd.DataFrame:
    """Read just the URL and label columns and convert to 1 = phishing."""
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nSee data/README.md for how to obtain it.")

    # utf-8-sig because the PhiUSIIL file starts with a byte-order mark,
    # which otherwise ends up glued to the first column name.
    df = pd.read_csv(path, usecols=["URL", label_column], encoding="utf-8-sig")
    df = df.rename(columns={"URL": "url", label_column: "raw_label"}).dropna(subset=["url", "raw_label"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df["target"] = (df["raw_label"].astype(int) != legitimate_value).astype(int)
    return df[["url", "target"]]


def build_feature_table(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, dict]:
    """
    Turn URLs into a feature table and record what was removed along the
    way. The counts end up in the results JSON, so the cleaning steps can be
    reported in the methodology chapter.
    """
    stats = {"rows_loaded": int(len(df))}
    df = df.drop_duplicates(subset=["url"])
    stats["after_exact_url_dedup"] = int(len(df))

    rows, keys, targets = [], [], []
    invalid = 0
    for i, (url, target) in enumerate(zip(df["url"], df["target"]), start=1):
        try:
            parsed = parse_url(url)
            rows.append(extract_features(url, feature_set))
        except URLValidationError:
            # A handful of dataset rows are over 2048 characters or otherwise
            # not something the web form would accept. Train on the same
            # inputs the app can actually receive.
            invalid += 1
            continue
        # With domain features, every URL on the same host produces the same
        # feature row. If one copy lands in train and another in test, the
        # test score is inflated, so the host is the unit of de-duplication.
        keys.append(parsed.host_no_www if feature_set == "domain" else url)
        targets.append(target)
        if i % 50_000 == 0:
            print(f"  features extracted for {i:,} / {len(df):,} URLs")
    stats["invalid_urls_dropped"] = invalid

    table = pd.DataFrame(rows, columns=FEATURE_SETS[feature_set])
    table["_key"] = keys
    table["target"] = targets

    # A host labelled both phishing and legitimate (e.g. a compromised site)
    # gives the model contradictory examples, so drop those hosts altogether.
    labels_per_key = table.groupby("_key")["target"].nunique()
    contradictory = set(labels_per_key[labels_per_key > 1].index)
    stats["conflicting_keys_dropped"] = len(contradictory)

    table = (table[~table["_key"].isin(contradictory)]
             .drop_duplicates(subset=["_key"])
             .drop(columns="_key")
             .reset_index(drop=True))

    stats["rows_after_cleaning"] = int(len(table))
    stats["phishing_rows"] = int(table["target"].sum())
    stats["legitimate_rows"] = int((table["target"] == 0).sum())
    return table, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the PhisScan URL models")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="domain")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tune-sample", type=int, default=40_000,
                        help="training rows used for the CV grid search (0 = all of them)")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-fpr", type=float, default=0.05,
                        help="highest acceptable CV false-positive rate when choosing a model")
    parser.add_argument("--quick", action="store_true", help="small grids and a 30k-row sample")
    parser.add_argument("--deploy", action="store_true",
                        help="also deploy a --feature-set full model (not recommended, see url_features.py)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.time()
    # Results for the experiment are saved under a separate prefix so they
    # never overwrite the deployed model's results.
    prefix = "url" if args.feature_set == "domain" else f"url_{args.feature_set}"

    print(f"Loading {args.data} ...")
    raw = load_dataset(args.data)
    if args.quick:
        raw = raw.sample(n=min(30_000, len(raw)), random_state=ev.RANDOM_STATE)

    print(f"Extracting '{args.feature_set}' features ...")
    table, stats = build_feature_table(raw, args.feature_set)
    print(f"  {stats}")

    feature_names = FEATURE_SETS[args.feature_set]
    X = table[feature_names].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=ev.RANDOM_STATE)

    # Grid-searching on all ~175k training rows is slow for the forest
    # models, so the search runs on a 40k stratified sample of the training
    # data. The chosen settings are then refitted on the whole training set.
    # Use --tune-sample 0 to search on everything if time allows.
    if args.tune_sample and len(X_train) > args.tune_sample:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=args.tune_sample, stratify=y_train, random_state=ev.RANDOM_STATE)
    else:
        X_tune, y_tune = X_train, y_train
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  CV tuning sample: {len(X_tune):,}")

    results, fitted = [], {}
    skipped = [] if ev.XGBOOST_AVAILABLE else ["XGBoost (package not installed: pip install xgboost)"]

    for name, (estimator, grid) in ev.candidate_models("url", quick=args.quick).items():
        steps = [("clf", estimator)]
        if name in NEEDS_SCALING:
            steps.insert(0, ("scale", StandardScaler()))
        pipeline = Pipeline(steps)
        grid = {f"clf__{param}": values for param, values in grid.items()}

        print(f"\nTraining {name} ...")
        model, result = ev.tune_and_evaluate(name, pipeline, grid, X_tune, y_tune, X_train, y_train,
                                             X_test, y_test, cv_folds=args.cv_folds)
        print(f"  CV F1 {result['cv_f1_mean']:.4f} | test F1 {result['f1']:.4f} | "
              f"test FPR {result['false_positive_rate']:.4f} | params {result['best_params']}")
        results.append(result)
        fitted[name] = model

    # The ensemble joins the comparison as one more candidate. It's built
    # from the tuned members above, so it costs no extra training.
    members = [r for r in results if r["model"] in ENSEMBLE_MEMBERS]
    if len(members) >= 2:
        print(f"\nEvaluating {ENSEMBLE_NAME} of {len(members)} models ...")
        results.append(ev.ensemble_result(ENSEMBLE_NAME, members, y_tune, y_test, args.cv_folds))
        fitted[ENSEMBLE_NAME] = SoftVotingEnsemble({r["model"]: fitted[r["model"]] for r in members})

    chosen, rule = ev.select_model(results, args.max_fpr)
    print()
    ev.print_table(results, chosen["model"])
    print("\nSelection rule:", rule)

    model = fitted[chosen["model"]]
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    version = f"url-{chosen['model'].lower().replace(' ', '_')}-{datetime.now():%Y%m%d%H%M}"

    ev.save_results(prefix, {
        "task": "url",
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "dataset": {"file": args.data.name, **stats},
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "cv_tuning_sample": int(len(X_tune)),
                  "cv_folds": args.cv_folds, "test_size": args.test_size, "random_state": ev.RANDOM_STATE},
        "selected_model": chosen["model"],
        "selection_rule": rule,
        "model_version": version,
        "trained_at": trained_at,
        "skipped_models": skipped,
        "selected_feature_importance": _global_importance(model, feature_names),
        "plots": ev.save_plots(prefix, results, y_test),
        "quick_mode": bool(args.quick),
        "sklearn_version": sklearn.__version__,
    }, results)

    if args.feature_set == "domain" or args.deploy:
        bundle = {
            # "pipeline" is the selected model (possibly the ensemble) and
            # decides the headline verdict. "members" holds every individual
            # algorithm so the report can show all of their verdicts. The
            # ensemble refers to the same member objects, so they're only
            # stored once in the file.
            "pipeline": model,
            "members": {r["model"]: fitted[r["model"]] for r in members},
            "model_name": chosen["model"],
            "model_version": version,
            "feature_set": args.feature_set,
            "feature_names": feature_names,
            # The "typical legitimate" values used by the per-URL explanation
            # in url_predictor.py. Medians rather than means, so a few
            # extreme domains don't drag them around.
            "reference_values": np.median(X_train[y_train == 0], axis=0).tolist(),
            "threshold": 0.5,
            "trained_at": trained_at,
            "test_metrics": {k: chosen[k] for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate")},
            "quick_mode": bool(args.quick),
            "label_map": {0: "Legitimate", 1: "Phishing"},
        }
        ev.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, ev.MODELS_DIR / "url_model.joblib", compress=3)
        print(f"\nSaved deployed model: saved_models/url_model.joblib ({chosen['model']})")
    else:
        print("\nExperiment results saved; the deployed model was not changed.")

    print(f"Results: research_outputs/model_evaluation/{prefix}_*  |  total time {time.time() - started:.0f}s")


def _global_importance(model, feature_names) -> dict | None:
    """Feature importance (trees) or |coefficient| (linear) for the admin page."""
    if isinstance(model, SoftVotingEnsemble):
        return None   # there's no single importance vector for an average of five models
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return dict(zip(feature_names, map(float, clf.feature_importances_)))
    if hasattr(clf, "coef_"):
        return dict(zip(feature_names, map(float, np.abs(clf.coef_[0]))))
    return None   # calibrated SVM: no single coefficient vector to report


if __name__ == "__main__":
    main()
