"""Metric helpers with definitions that the tests check by hand."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def relative_error(estimate: float, truth: float) -> float:
    if truth == 0:
        raise ValueError("relative error is undefined when truth is 0")
    return float(abs(estimate - truth) / abs(truth))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error. Rows are equally weighted."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when an actual value is 0")
    return float(np.mean(np.abs(actual - pred) / np.abs(actual)))


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    actual = np.asarray(y_true)
    if len(np.unique(actual)) < 2:
        raise ValueError("AUROC needs both classes")
    return float(roc_auc_score(actual, np.asarray(y_score, dtype=float)))
