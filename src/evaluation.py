"""
Evaluation Utilities
====================
Metrics computation, curve generation, and threshold analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute all evaluation metrics for binary classification.

    Returns
    -------
    metrics : dict
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return metrics


def compute_curves(y_true, y_prob):
    """
    Compute ROC, PR, and calibration curves.

    Returns
    -------
    curves : dict with keys 'roc', 'pr', 'calibration'
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)
    cal_prob_true, cal_prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    curves = {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": prec, "recall": rec, "thresholds": pr_thresholds},
        "calibration": {"prob_true": cal_prob_true, "prob_pred": cal_prob_pred},
    }
    return curves


def threshold_analysis(y_true, y_prob, thresholds=None):
    """
    Compute precision, recall, F1 at various thresholds.

    Returns
    -------
    df : DataFrame with columns [threshold, precision, recall, f1, specificity]
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.01)

    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results.append({
            "threshold": round(t, 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "specificity": spec,
        })

    return pd.DataFrame(results)


def find_optimal_threshold(y_true, y_prob, strategy="f1"):
    """
    Find the optimal classification threshold.

    Parameters
    ----------
    strategy : str
        'f1' maximizes F1-score.
        'recall_90' finds threshold where recall >= 0.90 with best precision.

    Returns
    -------
    optimal_threshold : float
    """
    df = threshold_analysis(y_true, y_prob)

    if strategy == "f1":
        idx = df["f1"].idxmax()
        return df.loc[idx, "threshold"]
    elif strategy == "recall_90":
        eligible = df[df["recall"] >= 0.90]
        if eligible.empty:
            return df.loc[df["recall"].idxmax(), "threshold"]
        idx = eligible["precision"].idxmax()
        return eligible.loc[idx, "threshold"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
