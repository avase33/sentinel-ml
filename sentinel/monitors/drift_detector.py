# Drift detector -- 2026-07-11 15:28:25
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DriftReport:
    feature: str
    reference_mean: float
    current_mean: float
    drift_score: float
    p_value: float
    is_drifted: bool
    severity: str
    detected_at: datetime

class DriftDetector:
    def __init__(self, threshold: float = 2.0, p_value_threshold: float = 0.05):
        self.threshold = threshold
        self.p_value_threshold = p_value_threshold
        self.reference_data: Optional[np.ndarray] = None

    def fit(self, reference: List[float]) -> None:
        self.reference_data = np.array(reference)

    def detect(self, current: List[float], feature_name: str = "feature") -> DriftReport:
        if self.reference_data is None:
            raise ValueError("Call fit() before detect()")
        ref = self.reference_data
        cur = np.array(current)
        drift_score = float(abs(cur.mean() - ref.mean()) / (ref.std() + 1e-8))
        _, p_value = stats.ks_2samp(ref, cur)
        is_drifted = drift_score > self.threshold or p_value < self.p_value_threshold
        severity = "critical" if drift_score > 4.0 else "high" if drift_score > 3.0 else "medium" if drift_score > 2.0 else "low"
        return DriftReport(
            feature=feature_name,
            reference_mean=float(ref.mean()),
            current_mean=float(cur.mean()),
            drift_score=drift_score,
            p_value=float(p_value),
            is_drifted=is_drifted,
            severity=severity,
            detected_at=datetime.utcnow(),
        )

    def detect_batch(self, current_data: Dict[str, List[float]]) -> List[DriftReport]:
        return [self.detect(values, feature) for feature, values in current_data.items()]