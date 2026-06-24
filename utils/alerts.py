"""
Alert management utilities for Sentinel-ML.
Defines alert severity levels and helper functions for creating and filtering alerts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a model monitoring alert."""

    model_name: str
    metric: str
    message: str
    severity: Severity
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    resolved: bool = False


def create_alert(
    model_name: str,
    metric: str,
    value: float,
    threshold: float,
    severity: Severity = Severity.MEDIUM,
) -> Alert:
    """
    Factory function to create a new Alert.

    Args:
        model_name: Name of the affected model.
        metric: Metric name that triggered the alert (e.g. 'accuracy', 'psi').
        value: Current observed metric value.
        threshold: The threshold that was breached.
        severity: Alert severity level.

    Returns:
        A new Alert instance.
    """
    message = (
        f"[{severity.value.upper()}] {model_name}: {metric} = {value:.4f} "
        f"exceeded threshold {threshold:.4f}"
    )
    return Alert(
        model_name=model_name,
        metric=metric,
        message=message,
        severity=severity,
        value=value,
        threshold=threshold,
    )


def filter_alerts(alerts: List[Alert], severity: Severity | None = None, resolved: bool | None = None) -> List[Alert]:
    """
    Filter a list of alerts by severity and/or resolution status.

    Args:
        alerts: List of Alert objects to filter.
        severity: If provided, only return alerts of this severity.
        resolved: If provided, filter by resolved status.

    Returns:
        Filtered list of Alert objects.
    """
    result = alerts
    if severity is not None:
        result = [a for a in result if a.severity == severity]
    if resolved is not None:
        result = [a for a in result if a.resolved == resolved]
    return result
