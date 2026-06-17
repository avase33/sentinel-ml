"""
/api/v1/features  — Online feature store endpoints
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.feature_store.store import FeatureStore

router = APIRouter()
store = FeatureStore()


class FeatureUpsertRequest(BaseModel):
    entity_id: str
    feature_set: str
    features: dict[str, Any]
    ttl: int = 86_400


class BatchUpsertRequest(BaseModel):
    rows: list[FeatureUpsertRequest]


@router.get("/features/{entity_id}", summary="Get online features")
async def get_features(entity_id: str, feature_set: str):
    """
    Retrieve online features for a single entity from Redis.
    Returns empty dict if entity not found (caller decides how to handle).
    """
    features = await store.get_online_features(entity_id=entity_id, feature_set=feature_set)
    return {"entity_id": entity_id, "feature_set": feature_set, "features": features}


@router.post("/features/batch", summary="Upsert features (online + offline)")
async def upsert_features(payload: FeatureUpsertRequest):
    """Write features to both Redis (online) and PostgreSQL (offline) stores."""
    await store.upsert(
        entity_id=payload.entity_id,
        feature_set=payload.feature_set,
        features=payload.features,
        ttl=payload.ttl,
    )
    return {"status": "ok", "entity_id": payload.entity_id}


@router.post("/features/batch/multi", summary="Upsert multiple entities at once")
async def upsert_features_multi(payload: BatchUpsertRequest):
    """Bulk upsert for pipeline / Kafka consumer use cases."""
    for row in payload.rows:
        await store.upsert(
            entity_id=row.entity_id,
            feature_set=row.feature_set,
            features=row.features,
            ttl=row.ttl,
        )
    return {"status": "ok", "rows_written": len(payload.rows)}


@router.post("/features/mget", summary="Multi-get online features")
async def multi_get_features(entity_ids: list[str], feature_set: str):
    """Get online features for multiple entities in a single Redis call."""
    result = await store.get_online_features_batch(
        entity_ids=entity_ids,
        feature_set=feature_set,
    )
    return {"feature_set": feature_set, "features": result, "found": len(result)}
