"""
Online + Offline Feature Store

Online store  → Redis (sub-10ms reads for real-time inference)
Offline store → PostgreSQL (batch feature retrieval for training)

Feature sets are versioned schemas. An entity (e.g. user_id, merchant_id)
maps to a named feature set containing typed feature values.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL   = os.getenv("REDIS_URL",    "redis://redis:6379/0")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://mlplatform:mlplatform@postgres:5432/features")

# TTL for online features (24 hours by default — override per feature set)
DEFAULT_ONLINE_TTL = 86_400


class FeatureStore:
    """
    Async feature store with online (Redis) and offline (PostgreSQL) backends.

    Usage:
        store = FeatureStore()
        await store.init()

        # Write features (goes to both Redis and Postgres)
        await store.upsert("user_123", "fraud_v2", {"tx_count_24h": 12, "avg_amount_7d": 85.5})

        # Online read (Redis, sub-10ms)
        features = await store.get_online_features("user_123", "fraud_v2")

        # Offline read (Postgres, for training)
        rows = await store.get_offline_features(entity_ids=["user_1", "user_2"], feature_set="fraud_v2")
    """

    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._pg: asyncpg.Pool | None = None

    async def init(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._pg = await asyncpg.create_pool(POSTGRES_DSN, min_size=2, max_size=10)
        await self._ensure_schema()
        logger.info("FeatureStore initialized (Redis + PostgreSQL)")

    async def _ensure_schema(self) -> None:
        async with self._pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_store (
                    entity_id    TEXT        NOT NULL,
                    feature_set  TEXT        NOT NULL,
                    features     JSONB       NOT NULL,
                    event_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY  (entity_id, feature_set, event_time)
                );
                CREATE INDEX IF NOT EXISTS fs_entity_set ON feature_store (entity_id, feature_set);
            """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        entity_id: str,
        feature_set: str,
        features: dict[str, Any],
        ttl: int = DEFAULT_ONLINE_TTL,
        event_time: datetime | None = None,
    ) -> None:
        """Write features to both online (Redis) and offline (Postgres) stores."""
        await self._write_online(entity_id, feature_set, features, ttl)
        await self._write_offline(entity_id, feature_set, features, event_time or datetime.utcnow())

    async def _write_online(self, entity_id: str, feature_set: str, features: dict, ttl: int) -> None:
        key = self._online_key(entity_id, feature_set)
        pipeline = self._redis.pipeline()
        pipeline.set(key, json.dumps(features))
        pipeline.expire(key, ttl)
        await pipeline.execute()

    async def _write_offline(
        self,
        entity_id: str,
        feature_set: str,
        features: dict,
        event_time: datetime,
    ) -> None:
        async with self._pg.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO feature_store (entity_id, feature_set, features, event_time)
                VALUES ($1, $2, $3::jsonb, $4)
                ON CONFLICT (entity_id, feature_set, event_time)
                DO UPDATE SET features = EXCLUDED.features
                """,
                entity_id,
                feature_set,
                json.dumps(features),
                event_time,
            )

    # ------------------------------------------------------------------
    # Online read (real-time inference path)
    # ------------------------------------------------------------------

    async def get_online_features(self, entity_id: str, feature_set: str) -> dict[str, Any]:
        """
        Read features from Redis. Returns {} if entity not found.
        Intended for the real-time inference hot path — expect sub-10ms.
        """
        key = self._online_key(entity_id, feature_set)
        raw = await self._redis.get(key)
        if not raw:
            logger.debug("Online feature miss: %s/%s", entity_id, feature_set)
            return {}
        return json.loads(raw)

    async def get_online_features_batch(
        self,
        entity_ids: list[str],
        feature_set: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Multi-get: returns a dict of entity_id → features for all found entities.
        Missing entities are omitted from the result.
        """
        keys = [self._online_key(eid, feature_set) for eid in entity_ids]
        values = await self._redis.mget(*keys)
        result = {}
        for eid, raw in zip(entity_ids, values):
            if raw:
                result[eid] = json.loads(raw)
        return result

    # ------------------------------------------------------------------
    # Offline read (training / backfill path)
    # ------------------------------------------------------------------

    async def get_offline_features(
        self,
        entity_ids: list[str],
        feature_set: str,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Point-in-time correct feature retrieval from Postgres.
        Returns the latest feature snapshot for each entity as of `as_of`.
        """
        ts = as_of or datetime.utcnow()
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (entity_id)
                    entity_id,
                    features,
                    event_time
                FROM feature_store
                WHERE entity_id = ANY($1)
                  AND feature_set = $2
                  AND event_time <= $3
                ORDER BY entity_id, event_time DESC
                """,
                entity_ids,
                feature_set,
                ts,
            )
        return [
            {"entity_id": r["entity_id"], "event_time": r["event_time"], **json.loads(r["features"])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _online_key(entity_id: str, feature_set: str) -> str:
        return f"features:{feature_set}:{entity_id}"

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
        if self._pg:
            await self._pg.close()
