"""
/api/v1/ab  — A/B Testing (champion-challenger) router
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.ab_testing.router import ABRouter

router = APIRouter()
ab_router = ABRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ABPredictRequest(BaseModel):
    experiment: str = Field(..., example="fraud_model_v3_test")
    entity_id: str = Field(..., example="user_abc123")
    features: dict[str, Any]


class ABPredictResponse(BaseModel):
    request_id: str
    experiment: str
    variant: str           # "champion" | "challenger"
    model_version: str
    prediction: float
    latency_ms: float


class OutcomeRequest(BaseModel):
    request_id: str
    label: float           # ground-truth outcome (e.g. 1 = fraud, 0 = legit)


class ExperimentConfig(BaseModel):
    experiment: str
    champion_model: str
    challenger_model: str
    champion_traffic: float = Field(0.9, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ab/predict", response_model=ABPredictResponse)
async def ab_predict(payload: ABPredictRequest):
    """
    Route a prediction request to champion or challenger based on traffic split.
    entity_id is hashed to ensure the same user always hits the same variant
    within an experiment (sticky assignment).
    """
    try:
        result = await ab_router.predict(
            experiment=payload.experiment,
            entity_id=payload.entity_id,
            features=payload.features,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {exc}")

    return ABPredictResponse(**result)


@router.post("/ab/outcome")
async def record_outcome(payload: OutcomeRequest):
    """
    Record the ground-truth outcome for a prediction request.
    Used to compute lift metrics (precision, recall, F1) per variant.
    """
    await ab_router.record_outcome(
        request_id=payload.request_id,
        label=payload.label,
    )
    return {"status": "recorded"}


@router.get("/ab/{experiment}/stats")
async def experiment_stats(experiment: str):
    """
    Return live experiment statistics:
    - Sample sizes per variant
    - Precision / recall / F1 per variant
    - Statistical significance (p-value via two-proportion z-test)
    - Recommendation: promote challenger or keep champion
    """
    try:
        return await ab_router.get_stats(experiment)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/ab/configure")
async def configure_experiment(config: ExperimentConfig):
    """Create or update an experiment's traffic split configuration."""
    await ab_router.configure(
        experiment=config.experiment,
        champion_model=config.champion_model,
        challenger_model=config.challenger_model,
        champion_traffic=config.champion_traffic,
    )
    return {"status": "configured", "experiment": config.experiment}


@router.get("/ab/experiments")
async def list_experiments():
    """List all active and completed experiments."""
    return await ab_router.list_experiments()
