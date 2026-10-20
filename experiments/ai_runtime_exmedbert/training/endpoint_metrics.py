"""Dependency-light binary endpoint metrics and validation threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def _arrays(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray):
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.size == 0 or truth.size != scores.size:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite and between zero and one")
    if truth.min() == truth.max():
        raise ValueError("both classes are required")
    return truth, scores


def _threshold_curve(truth: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[order]
    sorted_scores = scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), truth.size - 1]
    true_positive = np.cumsum(sorted_truth)[endpoints].astype(np.float64)
    false_positive = (endpoints + 1).astype(np.float64) - true_positive
    return sorted_scores[endpoints], true_positive, false_positive


def select_youden_threshold(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> float:
    truth, scores = _arrays(labels, probabilities)
    thresholds, true_positive, false_positive = _threshold_curve(truth, scores)
    sensitivity = true_positive / truth.sum()
    specificity = 1.0 - false_positive / (truth.size - truth.sum())
    youden = sensitivity + specificity - 1.0
    best = np.flatnonzero(np.isclose(youden, youden.max(), rtol=0.0, atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def binary_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    truth, scores = _arrays(labels, probabilities)
    if not 0 <= threshold <= 1 or calibration_bins < 1:
        raise ValueError("invalid threshold or calibration bin count")
    _, true_positive_curve, false_positive_curve = _threshold_curve(truth, scores)
    positives = float(truth.sum())
    negatives = float(truth.size - truth.sum())
    tpr = true_positive_curve / positives
    fpr = false_positive_curve / negatives
    extended_tpr = np.r_[0.0, tpr]
    extended_fpr = np.r_[0.0, fpr]
    roc_auc = float(
        np.sum(np.diff(extended_fpr) * (extended_tpr[:-1] + extended_tpr[1:]) / 2.0)
    )
    recall = tpr
    precision = true_positive_curve / (true_positive_curve + false_positive_curve)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (truth == 1)))
    false_positive = int(np.sum(predicted & (truth == 0)))
    true_negative = int(np.sum(~predicted & (truth == 0)))
    false_negative = int(np.sum(~predicted & (truth == 1)))

    bin_index = np.minimum((scores * calibration_bins).astype(int), calibration_bins - 1)
    calibration_error = 0.0
    for index in range(calibration_bins):
        selected = bin_index == index
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float(scores[selected].mean()) - float(truth[selected].mean())
            )

    return {
        "examples": int(truth.size),
        "positive": int(positives),
        "negative": int(negatives),
        "prevalence": float(truth.mean()),
        "loss_brier": float(np.mean((scores - truth) ** 2)),
        "calibration_error": calibration_error,
        "auroc": roc_auc,
        "average_precision": average_precision,
        "threshold": float(threshold),
        "sensitivity": true_positive / int(positives),
        "specificity": true_negative / int(negatives),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
