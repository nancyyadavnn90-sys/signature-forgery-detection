"""
Evaluation metrics for signature verification.

Positive class = FORGED (label 1).

    FAR  (False Acceptance Rate) = forged signatures accepted as genuine / all forged
         = FN / (FN + TP) = 1 - recall(forged)          <- the costliest error (fraud passes)
    FRR  (False Rejection Rate)  = genuine signatures rejected as forged / all genuine
         = FP / (FP + TN)                               <- customer inconvenience
    EER  (Equal Error Rate)      = the error rate at the threshold where FAR == FRR;
         the standard single-number metric in signature-verification papers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def far_frr(y_true, y_pred) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    forged, genuine = y_true == 1, y_true == 0
    far = float((y_pred[forged] == 0).mean()) if forged.any() else np.nan
    frr = float((y_pred[genuine] == 1).mean()) if genuine.any() else np.nan
    return far, frr


def equal_error_rate(y_true, scores) -> tuple[float, float]:
    """Return (EER, threshold at which FAR ~= FRR)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    far = 1 - tpr            # forged accepted
    frr = fpr                # genuine rejected
    i = int(np.nanargmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def threshold_for_far(y_true, scores, target_far: float = 0.05) -> float:
    """Largest decision threshold whose FAR is <= target_far.

    Must be computed on TRAINING (out-of-fold) scores and then applied to the test set.
    A larger threshold means fewer signatures are called forged, so we pick the largest one
    that still keeps forgeries out - this keeps the false rejections as low as possible.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    candidates = np.unique(scores)[::-1]
    for t in candidates:
        far, _ = far_frr(y_true, (scores >= t).astype(int))
        if far <= target_far:
            return float(t)
    return float(candidates[-1])


def compute_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """All metrics from the synopsis for one model on one data split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    far, frr = far_frr(y_true, y_pred)
    eer, _ = equal_error_rate(y_true, scores)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "FAR": far,
        "FRR": frr,
        "EER": eer,
    }


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["True Genuine", "True Forged"],
                        columns=["Pred Genuine", "Pred Forged"])


def get_scores(model, X) -> np.ndarray:
    """Probability of 'forged' (falls back to decision_function if needed)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
