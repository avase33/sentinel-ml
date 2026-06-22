"""
Sentinel-ML -- Alert Configuration
Define thresholds for drift and performance degradation alerts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class DriftThreshold:
    """Thresholds for triggering drift alerts."""
    ks_statistic: float = 0.1
    psi_score: float = 0.2
    wasserstein_distance: float = 0.15
    chi2_p_value: float = 0.05
    feature_drift_fraction: float = 0.3


@dataclass
class PerformanceThreshold:
    """Thresholds for model performance degradation."""
    accuracy_drop: float = 0.05
    f1_drop: float = 0.05
    rmse_increase: float = 0.1
    latency_p99_ms: float = 500.0
    error_rate: float = 0.01


@dataclass
class AlertConfig:
    """Full alert configuration for a monitored model."""
    model_id: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.SLACK])
    drift: DriftThreshold = field(default_factory=DriftThreshold)
    performance: PerformanceThreshold = field(default_factory=PerformanceThreshold)
    cooldown_minutes: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "drift_thresholds": {
                "ks_statistic": self.drift.ks_statistic,
                "psi_score": self.drift.psi_score,
            },
            "performance_thresholds": {
                "accuracy_drop": self.performance.accuracy_drop,
                "latency_p99_ms": self.performance.latency_p99_ms,
            },
            "cooldown_minutes": self.cooldown_minutes,
            "enabled": self.enabled,
        }


CLASSIFICATION_CONFIG = AlertConfig(
    model_id="default-classifier",
    severity=AlertSeverity.HIGH,
    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
)

REGRESSION_CONFIG = AlertConfig(
    model_id="default-regressor",
    severity=AlertSeverity.MEDIUM,
    drift=DriftThreshold(wasserstein_distance=0.2),
    performance=PerformanceThreshold(rmse_increase=0.15),
)
