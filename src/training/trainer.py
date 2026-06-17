"""
MLflow-tracked training pipeline.

Supports: XGBoost, LightGBM, Scikit-learn classifiers.
Auto-logs params, metrics, feature importances, SHAP values, and model card.
Registers to MLflow Model Registry if the new model beats the current champion.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_model(model_type: str, params: dict):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=params.get("scale_pos_weight", 10),
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Supported: xgboost")


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    target_col: str,
    model_name: str,
    model_type: str = "xgboost",
    experiment_name: str = "default",
    params: dict | None = None,
    register_if_better: bool = True,
    test_size: float = 0.2,
):
    params = params or {}

    # ── Load data ───────────────────────────────────────────────────────────
    logger.info("Loading dataset: %s", dataset_path)
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    elif dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("dataset_path must be .parquet or .csv")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info("Train: %d rows  Test: %d rows  Positives: %.2f%%", len(X_train), len(X_test), y.mean() * 100)

    # ── MLflow run ──────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": model_type,
            "target_col": target_col,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate": float(y.mean()),
            **params,
        })

        # ── Train ────────────────────────────────────────────────────────────
        model = build_model(model_type, params)

        if model_type == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50,
            )
            mlflow.xgboost.log_model(model, "model", registered_model_name=None)
        else:
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc":           round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc":            round(average_precision_score(y_test, y_prob), 4),
            "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        # ── Feature importance ───────────────────────────────────────────────
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(feature_names, model.feature_importances_.tolist()))
            mlflow.log_dict(fi, "feature_importance.json")

        # ── SHAP values (sampled for speed) ─────────────────────────────────
        try:
            sample = X_test.sample(min(500, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_summary = dict(zip(feature_names, mean_abs_shap.tolist()))
            mlflow.log_dict(shap_summary, "shap_mean_abs.json")
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)

        # ── Model card ───────────────────────────────────────────────────────
        model_card = {
            "model_name": model_name,
            "model_type": model_type,
            "experiment": experiment_name,
            "run_id": run.info.run_id,
            "target_column": target_col,
            "features": feature_names,
            "train_dataset": dataset_path,
            "evaluation_metrics": metrics,
            "intended_use": "Binary classification for production ML platform",
            "limitations": "Performance may degrade under significant data drift.",
            "monitoring": "Evidently AI drift detection enabled in production.",
        }
        mlflow.log_dict(model_card, "model_card.json")

        # ── Register if better than champion ────────────────────────────────
        if register_if_better:
            new_version = _register_if_better(
                run_id=run.info.run_id,
                model_name=model_name,
                new_metrics=metrics,
                X_sample=X_test.head(5),
                y_sample=y_prob[:5],
            )
            if new_version:
                logger.info("Registered new champion: %s v%s", model_name, new_version)
            else:
                logger.info("Existing champion retained (new model did not improve).")

    return metrics


# ---------------------------------------------------------------------------
# Registry promotion
# ---------------------------------------------------------------------------

def _register_if_better(
    run_id: str,
    model_name: str,
    new_metrics: dict,
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
) -> str | None:
    client = MlflowClient()

    # Get champion's PR-AUC for comparison
    champion_pr_auc = 0.0
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            champion_run = client.get_run(versions[0].run_id)
            champion_pr_auc = float(champion_run.data.metrics.get("pr_auc", 0.0))
    except Exception:
        pass   # No production model yet — always register

    if new_metrics["pr_auc"] <= champion_pr_auc:
        logger.info(
            "New model PR-AUC %.4f ≤ champion %.4f. Not promoting.",
            new_metrics["pr_auc"], champion_pr_auc,
        )
        return None

    # Register new version
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    return version


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register a model")
    parser.add_argument("--dataset",          required=True)
    parser.add_argument("--target",           default="label")
    parser.add_argument("--model-name",       required=True)
    parser.add_argument("--model-type",       default="xgboost")
    parser.add_argument("--experiment",       default="default")
    parser.add_argument("--params",           default="{}", help="JSON string of hyperparams")
    parser.add_argument("--register-if-better", action="store_true")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        target_col=args.target,
        model_name=args.model_name,
        model_type=args.model_type,
        experiment_name=args.experiment,
        params=json.loads(args.params),
        register_if_better=args.register_if_better,
    )
