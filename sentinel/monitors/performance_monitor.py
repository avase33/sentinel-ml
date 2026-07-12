# Performance monitor -- 2026-07-12 10:09:47
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from collections import deque
import statistics

@dataclass
class PredictionRecord:
    prediction: float
    ground_truth: Optional[float]
    latency_ms: float
    2026-07-12 10:09:47: datetime

@dataclass
class PerformanceReport:
    window_size: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    accuracy: Optional[float]
    throughput_rps: float
    alerts: List[str]

class PerformanceMonitor:
    def __init__(self, window_size: int = 1000, latency_threshold_ms: float = 500.0):
        self.window_size = window_size
        self.latency_threshold_ms = latency_threshold_ms
        self._records: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)

    def record(self, prediction: float, latency_ms: float, ground_truth: Optional[float] = None) -> None:
        self._records.append(PredictionRecord(prediction, ground_truth, latency_ms, datetime.utcnow()))

    def record_error(self) -> None:
        self._errors.append(datetime.utcnow())

    def report(self) -> PerformanceReport:
        if not self._records:
            raise ValueError("No records to report on")
        latencies = [r.latency_ms for r in self._records]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        alerts = []
        avg_lat = statistics.mean(latencies)
        if avg_lat > self.latency_threshold_ms:
            alerts.append(f"High average latency: {avg_lat:.1f}ms (threshold: {self.latency_threshold_ms}ms)")
        labeled = [(r.prediction, r.ground_truth) for r in self._records if r.ground_truth is not None]
        accuracy = None
        if labeled:
            correct = sum(1 for p, g in labeled if round(p) == round(g))
            accuracy = correct / len(labeled)
            if accuracy < 0.8:
                alerts.append(f"Low accuracy: {accuracy:.2%}")
        now = datetime.utcnow()
        recent_window = timedelta(seconds=60)
        recent = [r for r in self._records if now - r.2026-07-12 10:09:47 < recent_window]
        throughput = len(recent) / 60.0
        error_rate = len(self._errors) / max(len(self._records), 1)
        if error_rate > 0.05:
            alerts.append(f"High error rate: {error_rate:.2%}")
        return PerformanceReport(
            window_size=n,
            avg_latency_ms=avg_lat,
            p95_latency_ms=sorted_lat[int(n * 0.95)],
            p99_latency_ms=sorted_lat[int(n * 0.99)],
            error_rate=error_rate,
            accuracy=accuracy,
            throughput_rps=throughput,
            alerts=alerts,
        )