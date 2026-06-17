"""
/api/v1/predict  — Real-time model inference router
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.feature_store.store import FeatureStore
from src.serving.predictor import Predictor

router = APIRouter()
feature_store = FeatureStore()
predictor = Predictor()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_name: str = Field(..., example="fraud_detector")
    features: dict[str, Any] = Field(..., example={"amount": 150.0, "hour": 23})
    entity_id: str | None = Field(None, description="If set, merges online features from the feature store")
    version: str = Field("champion", description="'champion', 'challenger', or exact version string")
    feature_set: str | None = Field(None, description="Feature set name to pull from online store")


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    variant: str
    prediction: float | list[float]
    probabilities: dict[str, float] | None = None
    latency_ms: float


class BatchPredictRequest(BaseModel):
    model_name: str
    rows: list[dict[str, Any]]
    version: str = "champion"


class BatchPredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    predictions: list[float]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request):
    """
    Real-time single-row inference.

    - Optionally enriches request features with online feature store values.
    - Routes to champion or challenger model based on `version`.
    - Returns prediction + probability breakdown for classifiers.
    """
    start = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Enrich with online features if entity_id provided
    features = dict(payload.features)
    if payload.entity_id and payload.feature_set:
        online_features = await feature_store.get_online_features(
            entity_id=payload.entity_id,
            feature_set=payload.feature_set,
        )
        features = {**online_features, **features}   # request features override store

    try:
        result = await predictor.predict(
            model_name=payload.model_name,
            features=features,
            version=payload.version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        request_id=request_id,
        model_name=payload.model_name,
        model_version=result["version"],
        variant=result["variant"],
        prediction=result["prediction"],
        probabilities=result.get("probabilities"),
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(payload: BatchPredictRequest, request: Request):
    """
    Batch inference — returns one prediction per row.
    Backed by the same model registry; uses vectorized model.predict() under the hood.
    """
    start = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        result = await predictor.predict_batch(
            model_name=payload.model_name,
            rows=payload.rows,
            version=payload.version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        request_id=request_id,
        model_name=payload.model_name,
        model_version=result["version"],
        predictions=result["predictions"],
        latency_ms=round(latency_ms, 2),
    )


@router.get("/models", tags=["Model Inference"])
async def list_models():
    """List all loaded models and their active versions."""
    return await predictor.list_loaded_models()


@router.get("/models/{model_name}/versions", tags=["Model Inference"])
async def list_model_versions(model_name: str):
    """List all registered versions for a model from MLflow registry."""
    return await predictor.get_model_versions(model_name)
