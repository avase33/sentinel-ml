# Alert system -- 2026-07-12 14:23:23
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Callable
from enum import Enum
import json

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    model_id: str
    metric: str
    value: float
    threshold: float
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertManager:
    def __init__(self):
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        self._handlers.append(handler)

    def fire(self, title: str, message: str, severity: AlertSeverity,
             model_id: str, metric: str, value: float, threshold: float) -> Alert:
        import uuid
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            title=title, message=message, severity=severity,
            model_id=model_id, metric=metric, value=value, threshold=threshold,
            created_at=datetime.utcnow(),
        )
        self._alerts.append(alert)
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass
        return alert

    def resolve(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id and not a.resolved:
                a.resolved = True
                a.resolved_at = datetime.utcnow()
                return True
        return False

    def active(self, model_id: Optional[str] = None) -> List[Alert]:
        alerts = [a for a in self._alerts if not a.resolved]
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        return alerts

    def to_json(self) -> str:
        return json.dumps([{
            "alert_id": a.alert_id, "title": a.title, "severity": a.severity,
            "model_id": a.model_id, "metric": a.metric, "value": a.value,
            "resolved": a.resolved, "created_at": a.created_at.isoformat(),
        } for a in self._alerts], indent=2)

alert_manager = AlertManager()