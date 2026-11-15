"""
Small analysis helpers used in the notebook (EDA statistics and feature selection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def image_path(row) -> str:
    """Absolute path of an image from its file name and label."""
    folder = config.GENUINE_DIR if int(row["label"]) == config.LABEL_GENUINE else config.FORGED_DIR
    return str(folder / row["file"])


def eta_squared(x: pd.Series, groups: pd.Series) -> float:
    """Share of a feature's variance explained by a grouping (one-way ANOVA eta^2).

    eta^2(writer) close to 1 -> the feature mostly tells WHO wrote the signature.
    eta^2(class)  close to 1 -> the feature mostly tells genuine vs forged.
    """
    x = pd.Series(np.asarray(x, dtype=float))
    g = pd.Series(np.asarray(groups))
    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(xg) * (xg.mean() - grand) ** 2 for _, xg in x.groupby(g))
    return float(ss_between / ss_total)


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Standardised mean difference between forged (y=1) and genuine (y=0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    a, b = x[y == 1], x[y == 0]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def correlation_filter(X: pd.DataFrame, threshold: float = config.CORR_THRESHOLD):
    """Greedy filter: walk through the columns in order and drop a column if its absolute
    Pearson correlation with an already-kept column exceeds `threshold`.

    Fit on the TRAINING set only. Returns (kept_columns, dropped_table).
    """
    corr = X.corr().abs()
    kept, dropped = [], []
    for col in X.columns:
        if X[col].std() == 0:
            dropped.append({"feature": col, "correlated_with": "-", "abs_r": np.nan,
                            "reason": "constant"})
            continue
        partners = corr.loc[col, kept] if kept else pd.Series(dtype=float)
        if len(partners) and partners.max() > threshold:
            dropped.append({"feature": col, "correlated_with": partners.idxmax(),
                            "abs_r": round(float(partners.max()), 3),
                            "reason": f"|r| > {threshold}"})
        else:
            kept.append(col)
    return kept, pd.DataFrame(dropped, columns=["feature", "correlated_with", "abs_r", "reason"])


def same_value(a, b) -> bool:
    """Equality that treats 1 and 1.0 as equal (GridSearchCV may store mixed grids as floats)."""
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def param_curve(gs, param: str) -> pd.DataFrame:
    """Train / validation score of a GridSearchCV as ONE hyper-parameter varies,
    with every other hyper-parameter fixed at its best value."""
    res = pd.DataFrame(gs.cv_results_)
    mask = np.ones(len(res), dtype=bool)
    for k, v in gs.best_params_.items():
        if k == param:
            continue
        col = res[f"param_{k}"]
        mask &= np.array([same_value(x, v) for x in col], dtype=bool)
    sub = res.loc[mask, [f"param_{param}", "mean_train_roc_auc", "mean_test_roc_auc",
                         "std_test_roc_auc"]].copy()
    sub.columns = ["value", "train_auc", "val_auc", "val_std"]
    return sub.reset_index(drop=True)
