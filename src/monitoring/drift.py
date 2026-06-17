"""
Drift Monitor — data drift + prediction quality monitoring using Evidently AI.

Monitors:
  - Input feature distribution drift (PSI, KL-divergence, Wasserstein)
  - Prediction score drift
  - Data quality (missing values, schema violations)
  - Binary classification metrics vs. a baseline

On drift detection, fires a webhook to trigger automated retraining.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report

logger = logging.getLogger(__name__)

RETRAINING_WEBHOOK_URL = os.getenv("RETRAINING_WEBHOOK_URL", "")
DRIFT_REPORTS_DIR = Path(os.getenv("DRIFT_REPORTS_DIR", "/tmp/drift_reports"))
DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
FEATURE_DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_FEATURE_SHARE_THRESHOLD", "0.3"))
PREDICTION_DRIFT_THRESHOLD    = float(os.getenv("DRIFT_PRED_THRESHOLD", "0.1"))


@dataclass
class DriftReport:
    model_name: str
    drift_detected: bool
    feature_drift_share: float   # fraction of features that drifted
    drifted_features: list[str]
    prediction_drift: bool
    data_quality_issues: bool
    report_path: str
    summary: dict


class DriftMonitor:
    """
    Wraps Evidently AI to produce drift reports comparing a reference
    dataset (training distribution) against current serving data.

    Typical usage in a scheduled job or retraining trigger:

        monitor = DriftMonitor("fraud_detector", reference_df)
        report = monitor.run(current_df)
        if report.drift_detected:
            await monitor.trigger_retraining()
    """

    def __init__(
        self,
        model_name: str,
        reference_data: pd.DataFrame,
        target_col: str = "label",
        prediction_col: str = "score",
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
    ):
        self.model_name = model_name
        self.reference = reference_data
        self.target_col = target_col
        self.prediction_col = prediction_col

        feature_cols = [c for c in reference_data.columns if c not in {target_col, prediction_col}]
        self.column_mapping = ColumnMapping(
            target=target_col if target_col in reference_data.columns else None,
            prediction=prediction_col if prediction_col in reference_data.columns else None,
            numerical_features=numerical_features or [
                c for c in feature_cols if reference_data[c].dtype in ("float64", "int64")
            ],
            categorical_features=categorical_features or [
                c for c in feature_cols if reference_data[c].dtype == "object"
            ],
        )

    # ------------------------------------------------------------------
    # Core: run a full drift report
    # ------------------------------------------------------------------

    def run(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Compare current_data against the reference dataset.
        Returns a DriftReport with structured results and saves an HTML report to disk.
        """
        presets = [DataDriftPreset(), DataQualityPreset()]
        if self.target_col in current_data.columns and self.prediction_col in current_data.columns:
            presets.append(ClassificationPreset())

        report = Report(metrics=presets)
        report.run(
            reference_data=self.reference,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        result = report.as_dict()
        summary = self._parse_summary(result)

        # Save HTML report
        report_path = str(DRIFT_REPORTS_DIR / f"{self.model_name}_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S')}.html")
        report.save_html(report_path)
        logger.info("Drift report saved: %s", report_path)

        drift_detected = (
            summary["feature_drift_share"] >= FEATURE_DRIFT_SHARE_THRESHOLD
            or summary["prediction_drift"]
        )

        dr = DriftReport(
            model_name=self.model_name,
            drift_detected=drift_detected,
            feature_drift_share=summary["feature_drift_share"],
            drifted_features=summary["drifted_features"],
            prediction_drift=summary["prediction_drift"],
            data_quality_issues=summary["data_quality_issues"],
            report_path=report_path,
            summary=summary,
        )

        if drift_detected:
            logger.warning(
                "DRIFT DETECTED for '%s': %.0f%% of features drifted, prediction_drift=%s",
                self.model_name,
                dr.feature_drift_share * 100,
                dr.prediction_drift,
            )

        return dr

    # ------------------------------------------------------------------
    # Retraining webhook
    # ------------------------------------------------------------------

    async def trigger_retraining(self, model_name: str | None = None) -> None:
        """
        POST to the retraining webhook (GitHub Actions / Airflow / custom endpoint).
        Payload includes model name and trigger reason so the pipeline knows what to retrain.
        """
        url = RETRAINING_WEBHOOK_URL
        if not url:
            logger.warning("RETRAINING_WEBHOOK_URL not set — skipping trigger")
            return

        payload = {
            "model_name": model_name or self.model_name,
            "trigger": "drift_detected",
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("Retraining triggered for '%s': %s", self.model_name, resp.status_code)
            except Exception as exc:
                logger.error("Failed to trigger retraining: %s", exc)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_summary(evidently_result: dict) -> dict:
        """Extract structured drift metadata from Evidently's raw output dict."""
        summary: dict = {
            "feature_drift_share": 0.0,
            "drifted_features": [],
            "prediction_drift": False,
            "data_quality_issues": False,
        }
        try:
            metrics = evidently_result.get("metrics", [])
            for metric in metrics:
                metric_id = metric.get("metric", "")
                result_data = metric.get("result", {})

                if "DatasetDriftMetric" in metric_id:
                    summary["feature_drift_share"] = result_data.get("share_drifted_columns", 0.0)
                    summary["drifted_features"] = [
                        col for col, drifted in result_data.get("drift_by_columns", {}).items()
                        if drifted.get("drift_detected")
                    ]

                if "ColumnDriftMetric" in metric_id and "prediction" in metric_id.lower():
                    summary["prediction_drift"] = result_data.get("drift_detected", False)

                if "DataQualityMetric" in metric_id:
                    issues = result_data.get("current", {})
                    summary["data_quality_issues"] = (
                        issues.get("number_of_missing_values", 0) > 0
                        or issues.get("number_of_constant_columns", 0) > 0
                    )
        except Exception as exc:
            logger.warning("Could not fully parse Evidently result: %s", exc)
        return summary
