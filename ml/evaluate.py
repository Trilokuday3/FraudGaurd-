"""Evaluation metrics for the fraud classifier. PR-AUC is the headline metric
throughout this project -- never accuracy, which is meaningless at ~1.5%
prevalence."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_score, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict:
    metrics = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "thresholds": {},
    }
    scores = np.asarray(y_score)
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        metrics["thresholds"][t] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    return metrics


def calibration_curve_data(y_true, y_score, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    return calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
