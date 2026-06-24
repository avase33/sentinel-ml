"""
Metrics calculation utilities for Sentinel-ML.
Provides functions for evaluating model performance and detecting data drift.
"""

from __future__ import annotations
import math
from typing import Sequence


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Fraction of correct predictions in [0.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Mean Absolute Error between true and predicted values."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def population_stability_index(expected: Sequence[float], actual: Sequence[float], epsilon: float = 1e-8) -> float:
    """
    Compute the Population Stability Index (PSI) to detect data distribution drift.

    PSI < 0.1  → No significant change.
    PSI < 0.25 → Moderate change — monitor closely.
    PSI >= 0.25 → Significant shift — retrain recommended.

    Args:
        expected: Expected (baseline) proportions, must sum to 1.
        actual: Actual (current) proportions, must sum to 1.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI score.
    """
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of bins.")
    psi = 0.0
    for e, a in zip(expected, actual):
        e = max(e, epsilon)
        a = max(a, epsilon)
        psi += (a - e) * math.log(a / e)
    return psi


def calibration_error(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures how well predicted probabilities match observed frequencies.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of probability bins.

    Returns:
        ECE score in [0.0, 1.0] — lower is better.
    """
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = i * bin_size, (i + 1) * bin_size
        indices = [j for j, p in enumerate(y_prob) if low <= p < high]
        if not indices:
            continue
        avg_confidence = sum(y_prob[j] for j in indices) / len(indices)
        avg_accuracy = sum(y_true[j] for j in indices) / len(indices)
        ece += (len(indices) / n) * abs(avg_confidence - avg_accuracy)
    return ece
