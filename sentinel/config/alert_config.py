# Alert configuration -- 2026-07-10 10:22:34
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ThresholdConfig:
    drift_score_threshold: float = 2.0
    p_value_threshold: float = 0.05
    latency_p95_ms: float = 500.0
    latency_p99_ms: float = 1000.0
    error_rate_threshold: float = 0.05
    accuracy_drop_threshold: float = 0.10
    throughput_min_rps: float = 1.0

@dataclass
class NotificationConfig:
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    webhook_enabled: bool = False
    webhook_url: str = ""

@dataclass
class SentinelConfig:
    model_id: str
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    monitor_interval_seconds: int = 60
    reference_window_size: int = 1000
    current_window_size: int = 100
    enabled: bool = True

DEFAULT_CONFIG = SentinelConfig(
    model_id="default",
    thresholds=ThresholdConfig(
        drift_score_threshold=2.0,
        p_value_threshold=0.05,
        latency_p95_ms=500.0,
        error_rate_threshold=0.05,
    ),
)