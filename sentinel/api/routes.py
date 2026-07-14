# Sentinel API routes -- 2026-07-14 12:51:26
from fastapi import APIRouter, HTTPException
from typing import Optional
from sentinel.monitors.drift_detector import DriftDetector
from sentinel.monitors.performance_monitor import PerformanceMonitor
from sentinel.utils.alerts import alert_manager, AlertSeverity
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/sentinel", tags=["sentinel"])

_detectors: dict = {}
_monitors: dict = {}

class FitRequest(BaseModel):
    model_id: str
    reference_data: List[float]

class DetectRequest(BaseModel):
    model_id: str
    current_data: List[float]
    feature_name: str = "feature"

class RecordRequest(BaseModel):
    model_id: str
    prediction: float
    latency_ms: float
    ground_truth: Optional[float] = None

@router.post("/drift/fit")
def fit_detector(body: FitRequest):
    detector = DriftDetector()
    detector.fit(body.reference_data)
    _detectors[body.model_id] = detector
    return {"status": "fitted", "model_id": body.model_id, "samples": len(body.reference_data)}

@router.post("/drift/detect")
def detect_drift(body: DetectRequest):
    if body.model_id not in _detectors:
        raise HTTPException(status_code=404, detail="Detector not fitted for this model")
    report = _detectors[body.model_id].detect(body.current_data, body.feature_name)
    if report.is_drifted:
        alert_manager.fire(
            title=f"Data drift detected: {body.feature_name}",
            message=f"Drift score {report.drift_score:.2f} exceeded threshold",
            severity=AlertSeverity(report.severity),
            model_id=body.model_id,
            metric="drift_score",
            value=report.drift_score,
            threshold=2.0,
        )
    return {"is_drifted": report.is_drifted, "drift_score": report.drift_score, "p_value": report.p_value, "severity": report.severity}

@router.post("/performance/record")
def record_prediction(body: RecordRequest):
    if body.model_id not in _monitors:
        _monitors[body.model_id] = PerformanceMonitor()
    _monitors[body.model_id].record(body.prediction, body.latency_ms, body.ground_truth)
    return {"status": "recorded"}

@router.get("/performance/{model_id}")
def get_performance(model_id: str):
    if model_id not in _monitors:
        raise HTTPException(status_code=404, detail="No data for this model")
    report = _monitors[model_id].report()
    return report

@router.get("/alerts")
def get_alerts(model_id: Optional[str] = None):
    return alert_manager.active(model_id)