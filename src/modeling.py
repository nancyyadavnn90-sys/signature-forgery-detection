"""
Model zoo, hyper-parameter grids and the tuning / evaluation loop.

Every model is wrapped in a Pipeline with a StandardScaler where scaling matters
(Logistic Regression, SVM, KNN). The scaler is therefore re-fitted inside every
cross-validation fold, so no information from the validation fold leaks into training.
Tree-based models are scale-invariant and are used without a scaler.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .evaluation import compute_metrics, get_scores

RS = config.RANDOM_STATE

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",        # recall of the forged class = 1 - FAR
    "precision": "precision",
}
REFIT_METRIC = "roc_auc"       # threshold-independent and robust to the mild class imbalance


def scaled(estimator, name):
    return Pipeline([("scaler", StandardScaler()), (name, estimator)])


def model_zoo(fast: bool = False) -> dict:
    """name -> (estimator, param_grid, family). `fast=True` uses smaller grids."""
    C_grid = np.logspace(-3, 2, 6 if fast else 11)
    zoo = {
        # ------------------------------------------------ baseline + regularisation
        "LogReg (no penalty)": (
            scaled(LogisticRegression(C=np.inf, max_iter=5000), "lr"), {}, "Baseline"),
        "LogReg + L2 (Ridge)": (
            scaled(LogisticRegression(l1_ratio=0.0, max_iter=5000), "lr"),
            {"lr__C": C_grid}, "Baseline"),
        "LogReg + L1 (Lasso)": (
            scaled(LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=5000), "lr"),
            {"lr__C": C_grid}, "Baseline"),
        # ------------------------------------------------ single classifiers
        "SVM": (
            scaled(SVC(random_state=RS), "svm"),
            [{"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1] if fast else [0.01, 0.1, 1, 10]},
             {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10] if fast else [0.1, 1, 10, 100],
              "svm__gamma": ["scale", 0.01] if fast else ["scale", 0.001, 0.01, 0.1]}],
            "Single model"),
        "KNN": (
            scaled(KNeighborsClassifier(), "knn"),
            {"knn__n_neighbors": [1, 5, 11, 21] if fast else [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71],
             "knn__weights": ["uniform", "distance"],
             "knn__metric": ["euclidean", "manhattan"]},
            "Single model"),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RS),
            {"max_depth": [3, 5, 8, None] if fast else [2, 3, 4, 5, 6, 8, 10, None],
             "min_samples_leaf": [1, 10] if fast else [1, 5, 10, 20, 40],
             "criterion": ["gini", "entropy"],
             "ccp_alpha": [0.0, 0.005] if fast else [0.0, 0.002, 0.005, 0.01, 0.02]},
            "Single model"),
        # ------------------------------------------------ ensembles
        "Random Forest": (
            RandomForestClassifier(random_state=RS, n_jobs=1),
            {"n_estimators": [200] if fast else [100, 300, 500],
             "max_depth": [None, 10] if fast else [None, 8, 16],
             "max_features": ["sqrt", 0.5],
             "min_samples_leaf": [1, 3]},
            "Ensemble (bagging)"),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RS),
            {"n_estimators": [100, 300] if fast else [100, 200, 400],
             "learning_rate": [0.05, 0.1] if fast else [0.05, 0.1, 0.2],
             "max_depth": [2, 3] if fast else [2, 3, 4],
             "subsample": [0.8] if fast else [0.8, 1.0]},
            "Ensemble (boosting)"),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=RS),
                               random_state=RS),
            {"n_estimators": [100, 300] if fast else [100, 200, 400, 800],
             "learning_rate": [0.1, 1.0] if fast else [0.05, 0.1, 0.5, 1.0],
             "estimator__max_depth": [1, 2] if fast else [1, 2, 3]},
            "Ensemble (boosting)"),
    }
    return zoo


def tune_all(X, y, groups, zoo: dict, n_splits: int = config.CV_FOLDS,
             verbose: bool = True) -> dict:
    """GridSearchCV (GroupKFold by writer) for every model in the zoo."""
    cv = GroupKFold(n_splits=n_splits)
    searches = {}
    for name, (est, grid, _) in zoo.items():
        t0 = time.time()
        gs = GridSearchCV(est, grid, scoring=SCORING, refit=REFIT_METRIC, cv=cv,
                          n_jobs=-1, return_train_score=True, error_score="raise")
        gs.fit(X, y, groups=groups)
        searches[name] = gs
        if verbose:
            print(f"{name:<22s} CV AUC = {gs.best_score_:.4f}  "
                  f"({time.time() - t0:5.1f}s)  best = {gs.best_params_}")
    return searches


def finalize(searches: dict, X, y) -> dict:
    """Best estimator of every search; SVM is re-fitted with Platt probabilities."""
    final = {}
    for name, gs in searches.items():
        best = gs.best_estimator_
        if name == "SVM":
            best = clone(best).set_params(svm__probability=True).fit(X, y)
        final[name] = best
    return final


def add_voting(final: dict, X, y, members=("LogReg + L2 (Ridge)", "SVM",
                                           "Random Forest", "Gradient Boosting")) -> dict:
    """Soft-voting ensemble: averages the forged-probabilities of different model families."""
    vote = VotingClassifier([(m, clone(final[m])) for m in members], voting="soft", n_jobs=-1)
    vote.fit(X, y)
    final = dict(final)
    final["Soft Voting"] = vote
    return final


def cv_table(searches: dict, zoo: dict) -> pd.DataFrame:
    """Mean +/- std of the 5-fold cross-validation scores at the best parameters."""
    rows = []
    for name, gs in searches.items():
        i = gs.best_index_
        res = gs.cv_results_
        row = {"Model": name, "Family": zoo[name][2] if name in zoo else "Ensemble (voting)",
               "Best params": ", ".join(f"{k.split('__')[-1]}={v}"
                                        for k, v in gs.best_params_.items()) or "-"}
        for m in SCORING:
            row[f"CV {m}"] = res[f"mean_test_{m}"][i]
            row[f"CV {m} std"] = res[f"std_test_{m}"][i]
        row["Train AUC"] = res["mean_train_roc_auc"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def cv_scores_for(estimator, X, y, groups, n_splits: int = config.CV_FOLDS) -> pd.Series:
    """Out-of-fold forged-probabilities (writer-grouped CV) - used for threshold selection."""
    est = clone(estimator)
    method = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    oof = cross_val_predict(est, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
                            method=method, n_jobs=-1)
    return oof[:, 1] if oof.ndim == 2 else oof


def test_table(final: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for name, model in final.items():
        t0 = time.time()
        scores = get_scores(model, X_test)
        m = compute_metrics(y_test, scores, threshold)
        m["Model"] = name
        m["Predict time (ms)"] = (time.time() - t0) * 1000
        rows.append(m)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "FAR", "FRR", "EER",
            "Predict time (ms)"]
    return pd.DataFrame(rows)[cols]
