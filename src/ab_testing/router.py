"""
A/B Testing Router — Champion-Challenger traffic splitting with statistical significance tracking.

Design:
- Sticky assignment: entity_id is hashed to deterministically assign the same variant
  across requests within an experiment. No state needed for assignment itself.
- Outcome logging: stored in Redis as rolling counts per variant.
- Stats: two-proportion z-test for precision lift significance.
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from math import sqrt
from typing import Any

import redis.asyncio as aioredis
from scipy import stats

from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)

REDIS_URL = "redis://redis:6379/1"   # DB 1 reserved for A/B data


@dataclass
class ExperimentConfig:
    experiment: str
    champion_model: str
    challenger_model: str
    champion_traffic: float = 0.9   # fraction going to champion


class ABRouter:
    """
    Routes predictions to champion or challenger, records outcomes,
    and computes live statistical metrics.
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._predictor = Predictor()
        self._configs: dict[str, ExperimentConfig] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    async def configure(
        self,
        experiment: str,
        champion_model: str,
        challenger_model: str,
        champion_traffic: float = 0.9,
    ) -> None:
        cfg = ExperimentConfig(
            experiment=experiment,
            champion_model=champion_model,
            challenger_model=challenger_model,
            champion_traffic=champion_traffic,
        )
        self._configs[experiment] = cfg

        # Persist to Redis so configs survive restarts
        await self._redis.set(
            f"ab:config:{experiment}",
            json.dumps(cfg.__dict__),
        )
        logger.info("Experiment configured: %s", experiment)

    async def _load_config(self, experiment: str) -> ExperimentConfig:
        if experiment in self._configs:
            return self._configs[experiment]

        raw = await self._redis.get(f"ab:config:{experiment}")
        if not raw:
            raise KeyError(experiment)
        data = json.loads(raw)
        cfg = ExperimentConfig(**data)
        self._configs[experiment] = cfg
        return cfg

    # ------------------------------------------------------------------
    # Traffic splitting (deterministic hash-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_variant(entity_id: str, experiment: str, champion_traffic: float) -> str:
        """
        Hash entity_id + experiment name to a float in [0, 1).
        Values below champion_traffic → champion; else → challenger.
        Ensures the same user always hits the same variant.
        """
        key = f"{experiment}:{entity_id}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = (h % 10_000) / 10_000.0
        return "champion" if bucket < champion_traffic else "challenger"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict(
        self,
        experiment: str,
        entity_id: str,
        features: dict[str, Any],
    ) -> dict:
        cfg = await self._load_config(experiment)
        variant = self._assign_variant(entity_id, experiment, cfg.champion_traffic)
        model_name = cfg.champion_model if variant == "champion" else cfg.challenger_model

        result = await self._predictor.predict(
            model_name=model_name,
            features=features,
            version="champion",   # always use champion stage within each model
        )

        request_id = str(uuid.uuid4())
        ts = time.time()

        # Store request metadata for outcome recording
        await self._redis.setex(
            f"ab:req:{request_id}",
            86400,   # 24h TTL
            json.dumps({
                "experiment": experiment,
                "variant": variant,
                "model_name": model_name,
                "prediction": result["prediction"],
                "ts": ts,
            }),
        )

        # Increment prediction counter
        await self._redis.incr(f"ab:stats:{experiment}:{variant}:count")

        return {
            "request_id": request_id,
            "experiment": experiment,
            "variant": variant,
            "model_version": result["version"],
            "prediction": result["prediction"],
            "latency_ms": 0.0,   # filled by caller
        }

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    async def record_outcome(self, request_id: str, label: float) -> None:
        raw = await self._redis.get(f"ab:req:{request_id}")
        if not raw:
            logger.warning("record_outcome: request_id %s not found", request_id)
            return

        meta = json.loads(raw)
        experiment = meta["experiment"]
        variant = meta["variant"]
        prediction = meta["prediction"]

        # Binary classification metrics (prediction > 0.5 = positive)
        predicted_positive = prediction >= 0.5
        actual_positive = label >= 0.5

        if predicted_positive:
            if actual_positive:
                await self._redis.incr(f"ab:stats:{experiment}:{variant}:tp")
            else:
                await self._redis.incr(f"ab:stats:{experiment}:{variant}:fp")
        else:
            if actual_positive:
                await self._redis.incr(f"ab:stats:{experiment}:{variant}:fn")
            else:
                await self._redis.incr(f"ab:stats:{experiment}:{variant}:tn")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self, experiment: str) -> dict:
        cfg = await self._load_config(experiment)

        async def _fetch(variant: str) -> dict:
            keys = ["count", "tp", "fp", "fn", "tn"]
            values = await self._redis.mget(
                *[f"ab:stats:{experiment}:{variant}:{k}" for k in keys]
            )
            d = {k: int(v or 0) for k, v in zip(keys, values)}
            tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall
                else None
            )
            return {
                "variant": variant,
                "model": cfg.champion_model if variant == "champion" else cfg.challenger_model,
                "traffic_pct": cfg.champion_traffic if variant == "champion" else 1 - cfg.champion_traffic,
                "n_predictions": d["count"],
                "n_outcomes_recorded": tp + fp + fn + tn,
                "precision": round(precision, 4) if precision else None,
                "recall": round(recall, 4) if recall else None,
                "f1": round(f1, 4) if f1 else None,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        champ = await _fetch("champion")
        chall = await _fetch("challenger")

        p_value = self._significance_test(champ, chall)
        recommendation = self._recommend(champ, chall, p_value)

        return {
            "experiment": experiment,
            "champion": champ,
            "challenger": chall,
            "p_value": p_value,
            "significant_at_95": p_value < 0.05 if p_value else False,
            "recommendation": recommendation,
        }

    @staticmethod
    def _significance_test(champ: dict, chall: dict) -> float | None:
        """Two-proportion z-test on precision (TP / (TP+FP))."""
        try:
            n1 = champ["tp"] + champ["fp"]
            n2 = chall["tp"] + chall["fp"]
            if n1 < 30 or n2 < 30:
                return None   # not enough data
            p1 = champ["tp"] / n1
            p2 = chall["tp"] / n2
            p_pool = (champ["tp"] + chall["tp"]) / (n1 + n2)
            se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            return round(float(p_value), 4)
        except Exception:
            return None

    @staticmethod
    def _recommend(champ: dict, chall: dict, p_value: float | None) -> str:
        if p_value is None:
            return "collect_more_data"
        if p_value >= 0.05:
            return "keep_champion"
        cp = champ.get("precision") or 0
        dp = chall.get("precision") or 0
        return "promote_challenger" if dp > cp else "keep_champion"

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    async def list_experiments(self) -> list[dict]:
        keys = await self._redis.keys("ab:config:*")
        results = []
        for k in keys:
            raw = await self._redis.get(k)
            if raw:
                results.append(json.loads(raw))
        return results
